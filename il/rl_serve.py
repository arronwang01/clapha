"""Batched decisions for RL collection: one GPU process answers every running game's decisions.

One forward of our model costs about the same for 1 or 64 decisions (il/bench_forward.py: 33 ms at
batch 1, 0.6 ms per decision at batch 64 on the 4080), so collectors that each run their model on
one position at a time spend most of a game waiting on kernel launches. Here the models live in one
server process; a collector keeps everything else -- the engine, the console's rules, observations,
extras, decoding, rewards -- and sends each decision's model input (il/pack.py, one buffer) to the
server, which keeps every game's recurrent state and answers all waiting decisions of one model in
one forward. The collector's lane records exactly what the server's forward saw and chose.

    ./py -m il.rl_serve --run runs/rl/<run> [--address 127.0.0.1:26900] [--device cuda]

Models are named as firstlight_bot names them (fl:hog2, fl:general, clapha:v2), by checkpoint path,
or `latest`: the run's latest.pt, reloaded when the learner replaces it -- a game keeps the weights
it started with (its lane is one policy version).
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import threading
import time
from pathlib import Path

CLAPHA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAPHA / 'mac012'))

AUTHKEY = b'clapha-rl'
DEFAULT_ADDRESS = ('127.0.0.1', 26900)
_DYNAMIC = ('active_effects', 'relation_edges', 'candidates')
# CUDA graphs run at fixed shapes: every observation padded to these (measured in Hog 2.6 games,
# 2026-09-27: effects 0, relations max 46 (p99.9 43), candidates 6 always); a batch with a larger
# one runs eagerly. Batches are padded up to the next bucket.
GRAPH_CAPACITY = {'active_effects': 16, 'relation_edges': 64, 'candidates': 8}
GRAPH_BUCKETS = (8, 16, 32, 64)


def _address(text: str) -> tuple[str, int]:
    host, port = text.rsplit(':', 1)
    return host, int(port)


def _packed(record) -> tuple:
    from il.pack import pack
    structure, layout, data = pack(record)
    return structure, layout, bytes(data)


def _wire(stored) -> tuple:
    """A decision's model input as sent: (stored record, (key, structure, layout, bytes)). An input
    within GRAPH_CAPACITY is padded to it first, so every such input of a model has one byte layout
    (key); the server then stacks a batch by copying bytes and its graph unpacks the fields on the
    GPU. Anything larger goes unpadded with key None and runs eagerly."""
    import hashlib
    from il.train import _rows
    from native_runner.training.v4.cache import _pad_dynamic_observation
    fits = all(_rows(getattr(stored, name)) <= GRAPH_CAPACITY[name] for name in _DYNAMIC)
    if fits:
        stored = _pad_dynamic_observation(stored, active_effect_count=GRAPH_CAPACITY['active_effects'],
                                          relation_edge_count=GRAPH_CAPACITY['relation_edges'],
                                          candidate_count=GRAPH_CAPACITY['candidates'])
    structure, layout, data = _packed(stored)
    key = hashlib.sha1(structure + repr(layout).encode()).hexdigest()[:16] if fits else None
    return stored, (key, structure, layout, data)


def _unpacked(message: tuple):
    from il.pack import unpack
    structure, layout, data = message
    return unpack((structure, layout, bytearray(data)))


# ---- server ----------------------------------------------------------------------------------

class _Graph:
    """One captured sample_for_ppo_rollout at a fixed batch size, fed from one [batch, bytes] buffer:
    each row is one decision's input as il/pack laid it out (all rows share the layout). The graph
    itself cuts the fields out of the buffer and casts them, so a batch costs one host-to-device copy
    and one replay instead of ~200 small copies per decision (a forward is launch-bound: 33 ms at
    batch 1 or 64 run op by op)."""

    def __init__(self, module, structure: bytes, layout: list, size: int, device, pool, sample: bytes) -> None:
        import io
        import torch
        from il.pack import _Unpacker
        from native_runner.training.v4.tensors import RecurrentPolicyStateV4
        self.size = size
        width = len(sample)
        self.host = torch.empty((size, width), dtype=torch.uint8).pin_memory()
        self.host[:] = torch.frombuffer(bytearray(sample), dtype=torch.uint8)      # real rows to capture on
        self.flat = self.host.to(device)
        blank = module.initial_state(size, device=device)
        self.state = RecurrentPolicyStateV4(hidden=blank.hidden.clone(), cell=blank.cell.clone())
        self.first = torch.zeros(size, dtype=torch.bool, device=device)

        def forward():
            tensors = []
            for offset, length, dtype, shape in layout:
                rows = (size, *tuple(shape)[1:])
                if length == 0:
                    tensors.append(torch.zeros(rows, dtype=dtype, device=device))
                    continue
                field = self.flat[:, offset:offset + length].contiguous().view(dtype).reshape(rows)
                tensors.append(field.float() if field.is_floating_point() else field)
            batch = _Unpacker(io.BytesIO(structure), tensors).load()
            return module.sample_for_ppo_rollout(batch, self.state, episode_start=self.first, validate=False)

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self.output = forward()

    def run(self, rows: list, state, first):
        import numpy as np
        host = self.host.numpy()
        for index, data in enumerate(rows):
            host[index] = np.frombuffer(data, dtype=np.uint8)
        host[len(rows):] = host[0]                      # rows that only fill the bucket
        self.flat.copy_(self.host, non_blocking=True)
        self.state.hidden.copy_(state.hidden)
        self.state.cell.copy_(state.cell)
        self.first.copy_(first)
        self.graph.replay()
        return self.output


class _Model:
    def __init__(self, key: str, path: Path, device, graphs: bool = False) -> None:
        from il.extras import cache_static_encodings, checkpoint_extra, load_policy
        self.key, self.path, self.device = key, path, device
        extra = checkpoint_extra(path) if path.is_file() else {}
        self.version = int(extra.get('update', 0))
        self.module = load_policy(path, device)
        self.module.eval()
        cache_static_encodings(self.module)
        self.sessions = 0
        self.used = time.time()
        self.graphs: dict[tuple, _Graph] | None = {} if graphs else None
        self.pool = None

    def graph(self, key: str, structure: bytes, layout: list, size: int, sample: bytes) -> _Graph:
        import torch
        if (key, size) not in self.graphs:
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            started = time.perf_counter()
            with torch.inference_mode():
                self.graphs[(key, size)] = _Graph(self.module, structure, layout, size, self.device, self.pool, sample)
            print(f'{self.key}: captured batch {size} in {time.perf_counter() - started:.1f} s', flush=True)
        return self.graphs[(key, size)]


class Server:
    def __init__(self, run: Path | None, device, max_batch: int, graphs: bool = False) -> None:
        self.run, self.device, self.max_batch = run, device, max_batch
        self.use_graphs = graphs and device.type == 'cuda'
        self.models: dict[str, _Model] = {}          # key -> current weights for new games
        self.sessions: dict[str, dict] = {}          # session id -> model, recurrent state, first
        self.connections: list = []
        self.pending_connections: list = []
        self.lock = threading.Lock()
        self.latest_mtime = None
        self.loading: threading.Thread | None = None
        self.loaded: _Model | None = None
        self.stats = {'decisions': 0, 'forwards': 0, 'forward_ms': 0.0, 'since': time.time()}

    def _path(self, key: str) -> Path:
        import firstlight_bot as FLB
        if key == 'latest':
            if self.run is None:
                raise ValueError('latest needs --run')
            return self.run / 'latest.pt'
        return Path(FLB.CHECKPOINTS.get(key, Path(key)))

    def model(self, key: str) -> _Model:
        if key not in self.models:
            self._evict()
            path = self._path(key)
            self.models[key] = _Model(key, path, self.device, self.use_graphs)
            if key == 'latest':
                self.latest_mtime = path.stat().st_mtime
            print(f'loaded {key} ({path}, update {self.models[key].version})', flush=True)
        self.models[key].used = time.time()
        return self.models[key]

    def _evict(self, keep: int = 8) -> None:
        """League snapshots come and go: past `keep` models, drop the least recently used idle one."""
        idle = sorted((m.used, key) for key, m in self.models.items() if key != 'latest' and m.sessions == 0)
        while len(self.models) >= keep and idle:
            _used, key = idle.pop(0)
            del self.models[key]
            print(f'unloaded {key}', flush=True)

    def _watch_latest(self) -> None:
        """A replaced latest.pt is loaded in a thread; new games then get it, running ones finish
        with the weights they started with."""
        if 'latest' not in self.models:
            return
        if self.loaded is not None:
            self.models['latest'], self.loaded = self.loaded, None
            print(f'latest -> update {self.models["latest"].version}', flush=True)
            return
        if self.loading is not None and self.loading.is_alive():
            return
        path = self._path('latest')
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if mtime == self.latest_mtime:
            return
        self.latest_mtime = mtime

        def load() -> None:
            try:
                self.loaded = _Model('latest', path, self.device, self.use_graphs)
            except Exception as error:  # noqa: BLE001  (being replaced: the next change retries)
                print(f'latest reload failed: {type(error).__name__}: {error}', flush=True)
                self.latest_mtime = None
        self.loading = threading.Thread(target=load, daemon=True)
        self.loading.start()

    def _accept(self, listener) -> None:
        while True:
            connection = listener.accept()
            with self.lock:
                self.pending_connections.append(connection)

    def _drop(self, connection) -> None:
        self.connections.remove(connection)
        for sid in [sid for sid, s in self.sessions.items() if s['connection'] is connection]:
            self._close(sid)

    def _close(self, sid: str) -> None:
        session = self.sessions.pop(sid, None)
        if session is not None:
            session['model'].sessions -= 1

    def _forward(self, model: _Model, requests: list) -> None:
        import torch
        from il.train import _rows
        from native_runner.training.v4.cache import _pad_dynamic_observation
        from native_runner.training.v4.tensors import RecurrentPolicyStateV4, concatenate_tensor_records
        size = len(requests)
        keys = {request['payload'][0] for request in requests}
        bucket = next((b for b in GRAPH_BUCKETS if b >= size), None)
        graphed = model.graphs is not None and bucket is not None and len(keys) == 1 and None not in keys
        sessions = [self.sessions[request['sid']] for request in requests]
        started = time.perf_counter()
        with torch.inference_mode():
            hidden = [s['state'].hidden for s in sessions]
            cell = [s['state'].cell for s in sessions]
            flags = [s['first'] for s in sessions]
            output = None
            if graphed:
                if bucket > size:
                    blank = model.module.initial_state(bucket - size, device=self.device)
                    hidden, cell = hidden + [blank.hidden], cell + [blank.cell]
                state = RecurrentPolicyStateV4(hidden=torch.cat(hidden), cell=torch.cat(cell))
                first = torch.tensor(flags + [False] * (bucket - size), dtype=torch.bool, device=self.device)
                key, structure, layout, sample = requests[0]['payload']
                try:
                    graph = model.graph(key, structure, layout, bucket, sample)
                    output = graph.run([request['payload'][3] for request in requests], state, first)
                except Exception as error:  # noqa: BLE001  (not capturable: this model runs eagerly)
                    print(f'{model.key}: CUDA graph failed, eager from now on: {type(error).__name__}: {error}',
                          flush=True)
                    model.graphs, graphed = None, False
            if output is None:
                observations = [_unpacked(request['payload'][1:]) for request in requests]
                counts = {name: max(_rows(getattr(o, name)) for o in observations) for name in _DYNAMIC}
                batch = concatenate_tensor_records(tuple(
                    _pad_dynamic_observation(o, active_effect_count=counts['active_effects'],
                                             relation_edge_count=counts['relation_edges'],
                                             candidate_count=counts['candidates']) for o in observations))
                state = RecurrentPolicyStateV4(hidden=torch.cat(hidden[:size]), cell=torch.cat(cell[:size]))
                first = torch.tensor(flags, dtype=torch.bool, device=self.device)
                output = model.module.sample_for_ppo_rollout(batch.to_model_input(self.device), state,
                                                             episode_start=first, validate=False)
            actions = output.actions.to('cpu')
            log_prob = output.log_prob.float().reshape(-1).cpu().tolist()
            value = output.value.float().reshape(-1).cpu().tolist()
            next_hidden, next_cell = output.next_state.hidden, output.next_state.cell
            for row, (request, session) in enumerate(zip(requests, sessions)):
                session['state'] = RecurrentPolicyStateV4(hidden=next_hidden[row:row + 1].clone(),
                                                          cell=next_cell[row:row + 1].clone())
                session['first'] = False
                picked = actions.index_select(torch.tensor([row]))
                result = (_packed(picked), log_prob[row], value[row])
                group = request.get('group')
                if group is None:
                    request['connection'].send(('ok', *result))
                else:
                    group['results'][request['index']] = result
                    group['left'] -= 1
                    if group['left'] == 0:
                        group['connection'].send(('ok', group['results']))
        self.stats['forwards'] += 1
        self.stats['decisions'] += len(requests)
        self.stats['forward_ms'] += (time.perf_counter() - started) * 1000
        self.stats['graphed'] = self.stats.get('graphed', 0) + int(graphed)

    def _gpu_gb(self) -> float:
        import torch
        return torch.cuda.memory_reserved(self.device) / 2 ** 30 if self.device.type == 'cuda' else 0.0

    @staticmethod
    def _fail(request: dict, reason: str) -> None:
        group = request.get('group')
        if group is None:
            request['connection'].send(('error', reason))
        elif group['left'] > 0:
            group['left'] = 0           # the game's other results are dropped with it
            group['connection'].send(('error', reason))

    def _handle(self, connection, message, pending: list) -> None:
        kind = message[0]
        if kind == 'open':
            _, sid, key = message
            model = self.model(key)
            self.sessions[sid] = {'connection': connection, 'model': model, 'first': True,
                                  'state': model.module.initial_state(1, device=self.device)}
            model.sessions += 1
            module = model.module
            connection.send(('ok', model.version, (float(module.ppo_gate_temperature), float(module.ppo_action_temperature),
                                                   float(module.ppo_continue_temperature)),
                             '_extras_head' in module.__dict__))
        elif kind == 'decide':
            _, sid, payload = message
            pending.append({'sid': sid, 'connection': connection, 'payload': payload})
        elif kind == 'decide_many':
            # one game's sides together: one reply with every result, in order
            _, items = message
            group = {'connection': connection, 'results': [None] * len(items), 'left': len(items)}
            for index, (sid, payload) in enumerate(items):
                pending.append({'sid': sid, 'connection': connection, 'payload': payload,
                                'group': group, 'index': index})
        elif kind == 'close':
            self._close(message[1])
            connection.send(('ok',))
        else:
            connection.send(('error', f'unknown request {kind}'))

    def serve(self, address: tuple[str, int]) -> None:
        from multiprocessing.connection import Listener, wait
        listener = Listener(address, authkey=AUTHKEY)
        threading.Thread(target=self._accept, args=(listener,), daemon=True).start()
        print(f'serving on {address[0]}:{address[1]} ({self.device})', flush=True)
        pending: list = []
        last_report = time.time()
        while True:
            with self.lock:
                self.connections += self.pending_connections
                self.pending_connections = []
            # everything that has arrived; wait briefly only when there is nothing to do
            ready = wait(self.connections, timeout=0 if pending else 0.01) if self.connections else []
            if not self.connections:
                time.sleep(0.01)
            for connection in ready:
                try:
                    message = connection.recv()
                except (EOFError, OSError):
                    self._drop(connection)
                    continue
                try:
                    self._handle(connection, message, pending)
                except Exception as error:  # noqa: BLE001  (one bad request must not stop the others)
                    connection.send(('error', f'{type(error).__name__}: {error}'))
            if pending and not ready:
                by_model: dict[int, list] = {}
                for request in pending:
                    session = self.sessions.get(request['sid'])
                    if session is None:
                        self._fail(request, 'unknown session')
                        continue
                    by_model.setdefault(id(session['model']), []).append(request)
                pending = []
                for requests in by_model.values():
                    model = self.sessions[requests[0]['sid']]['model']
                    for first in range(0, len(requests), self.max_batch):
                        chunk = requests[first:first + self.max_batch]
                        try:
                            self._forward(model, chunk)
                        except Exception as error:  # noqa: BLE001
                            for request in chunk:
                                self._fail(request, f'{type(error).__name__}: {error}')
            self._watch_latest()
            if time.time() - last_report > 60:
                span = time.time() - self.stats['since']
                forwards = max(1, self.stats['forwards'])
                print(f"{self.stats['decisions'] / span:7.1f} decisions/s, {self.stats['decisions'] / forwards:5.1f} "
                      f"per forward, {self.stats['forward_ms'] / forwards:5.1f} ms per forward "
                      f"({self.stats.get('graphed', 0)} of {self.stats['forwards']} graphed), "
                      f"{len(self.sessions)} games' sides, {len(self.connections)} connections, "
                      f"{len(self.models)} models, GPU {self._gpu_gb():.1f} GB", flush=True)
                self.stats = {'decisions': 0, 'forwards': 0, 'forward_ms': 0.0, 'since': time.time()}
                last_report = time.time()


# ---- collector side ------------------------------------------------------------------------------

class Client:
    """One connection to the server (one per game thread / process)."""

    def __init__(self, address: tuple[str, int] = DEFAULT_ADDRESS) -> None:
        from multiprocessing.connection import Client as Connect
        for attempt in itertools.count():
            try:
                self.connection = Connect(address, authkey=AUTHKEY)
                break
            except ConnectionRefusedError:
                if attempt > 120:
                    raise
                time.sleep(1.0)
        self._ids = itertools.count()

    def _call(self, message):
        self.connection.send(message)
        reply = self.connection.recv()
        if reply[0] != 'ok':
            raise RuntimeError(f'inference server: {reply[1]}')
        return reply[1:]

    def open(self, key: str) -> tuple[str, int, tuple, bool | None]:
        """-> session id, the weights' update number, the sampling temperatures (for the lane), whether
        the served model has the extras head (None from an older server)."""
        sid = f'{os.getpid()}-{id(self)}-{next(self._ids)}'
        version, temperatures, *rest = self._call(('open', sid, key))
        return sid, int(version), tuple(temperatures), (bool(rest[0]) if rest else None)

    def decide(self, sid: str, payload: tuple) -> tuple:
        """payload: _wire()'s second part."""
        packed, log_prob, value = self._call(('decide', sid, payload))
        return _unpacked(packed), float(log_prob), float(value)

    def decide_many(self, items: list) -> list:
        """[(session id, payload)] -> [(actions, log-prob, value)], one round trip."""
        (results,) = self._call(('decide_many', list(items)))
        return [(_unpacked(packed), float(log_prob), float(value)) for packed, log_prob, value in results]

    def close(self, sid: str) -> None:
        self._call(('close', sid))


class RemoteSession:
    """PolicySessionV4's decide() with the forward on the server: the same tensorizer, extras,
    decoding and recorded action; every decision's input, action, log-prob and value kept."""

    def __init__(self, tensorizer, client: Client, key: str) -> None:
        self.tensorizer = tensorizer
        self.client, self.key = client, key
        self.next_extras = None
        self.sid, self.version, self.temperatures, self.model_has_extras = client.open(key)
        self.steps: list[dict] = []

    def end_episode(self) -> None:
        self.tensorizer.end_episode()
        self.client.close(self.sid)

    def prepare(self, observation) -> tuple:
        """The model input for this decision: (host batch, the stored input -- exactly what the server
        gets -- and its wire payload)."""
        import torch
        from il.extras import extend_batch
        host = self.tensorizer.tensorize(observation, validate=False)
        if self.next_extras is not None:
            host = extend_batch(host, self.next_extras)
        stored, payload = _wire(host.to_storage('cpu', float_dtype=torch.float16))
        return host, stored, payload

    def decide(self, observation):
        host, stored, payload = self.prepare(observation)
        started = time.perf_counter()
        actions, log_prob, value = self.client.decide(self.sid, payload)
        return self.finish(observation, host, stored, actions, log_prob, value,
                           (time.perf_counter() - started) * 1000)

    def finish(self, observation, host, stored, actions, log_prob: float, value: float, inference_ms: float = 0.0):
        """Decode the server's choice, feed it back to the tensorizer, keep the step for the lane."""
        from native_runner.training.v4.decoding import decode_action_sequence_v4
        from native_runner.training.v4.policy_session import PolicyDecisionV4
        tensorizer = self.tensorizer
        decoded = decode_action_sequence_v4(
            actions, host.candidates, row=0, observation=observation, catalog=tensorizer.catalog,
            deck=tensorizer.deck, card_costs=tensorizer.card_costs,
            ability_id_by_vocab_id=tensorizer.ability_id_by_vocab_id,
            horizontal_mirror=tensorizer.perspective.horizontal_mirror,
            hand_slot_permutation=tensorizer.perspective.hand_slot_permutation,
            config=tensorizer.config, base_latency_ticks=1, base_latency_ms=0.0, validate=False)
        tensorizer.record_action(actions, host, row=0, validate=False)
        self.steps.append({'observation': stored, 'action': actions, 'log_prob': log_prob, 'value': value})
        return PolicyDecisionV4(decoded=decoded, inference_ms=inference_ms)


_RECIPES: dict = {}


def _recipe(path: Path) -> dict:
    """A checkpoint's `extra` (recipe, extras head), read once per file version."""
    from il.extras import checkpoint_extra
    stamp = (str(path), path.stat().st_mtime)
    if stamp not in _RECIPES:
        extra = checkpoint_extra(path)
        # il.train's final checkpoints say extras=...; every checkpoint with the head carries
        # extras_version (il.rl_learn's did not say extras=: pilot1's learner played without its
        # extras inputs from update 1 on, 2026-09-29)
        _RECIPES[stamp] = {'recipe': extra.get('recipe', ''),
                           'extras': bool(extra.get('extras') or extra.get('extras_version'))}
    return _RECIPES[stamp]


def remote_runner(client: Client, key: str, checkpoint: str | None = None):
    """A FirstLightRunner whose decisions come from the server's `key` model. checkpoint: where to
    read the model's recipe (our inputs, extras) without loading it; default: the key's path."""
    import types
    import firstlight_bot as FLB

    class RemoteRunner(FLB.FirstLightRunner):
        def __init__(self) -> None:
            super().__init__(None)                       # no local model
            path = Path(checkpoint) if checkpoint else Path(FLB.CHECKPOINTS.get(key, Path(key)))
            extra = _recipe(path) if path.is_file() and not key.startswith('fl:') else {}
            self.model_name = key
            self.clapha_inputs = str(extra.get('recipe', '')).startswith('clapha')
            self.has_extras = bool(extra.get('extras'))
            self.elixir_lead = 9 if self.clapha_inputs else 0
            # il.duel feeds extras when the model has the head
            self.model = types.SimpleNamespace(**({'_extras_head': True} if self.has_extras else {}))
            self.version = 0

        def start_battle(self, *args, **kwargs) -> None:
            model, self.model = self.model, None             # the base class builds the tensorizer
            try:
                super().start_battle(*args, **kwargs)
            finally:
                self.model = model
            self.session = RemoteSession(self.session.tensorizer, client, key)
            self.version, self.temperatures = self.session.version, self.session.temperatures
            served = self.session.model_has_extras
            if served is not None and served != self.has_extras:
                # the served weights decide: a model with the head gets its extras inputs (pilot1's
                # learner played without them because its checkpoints' recipe did not say so)
                print(f'{key}: the served model {"has" if served else "has no"} extras head, the recipe said '
                      f'{self.has_extras}: following the model', flush=True)
                self.has_extras = served
                self.model = types.SimpleNamespace(**({'_extras_head': True} if served else {}))
            self.last_steps = []

        def end_battle(self) -> None:
            # il.duel ends every battle itself; the lane is read afterwards
            if isinstance(self.session, RemoteSession):
                self.last_steps = self.session.steps
            super().end_battle()

        @staticmethod
        def moves(decision) -> list:
            """FirstLightRunner.decide's moves from a decision."""
            moves = []
            for action in getattr(getattr(decision, 'decoded', None), 'actions', ()) or ():
                move = FLB.Move((action.kind, action.hand_slot, action.card_id, action.target_grid,
                                 int(getattr(action, 'execute_offset_ticks', 0) or 0)))
                move.source_entity = getattr(action, 'source_entity', None)
                move.ability_id = getattr(action, 'ability_id', None)
                moves.append(move)
            moves.sort(key=lambda move: move[4])
            return moves

    return RemoteRunner()


def decide_many(ready: list) -> dict:
    """il.duel.play_match's decide_many: [(side, remote runner, observation)] -> {side: moves}, both
    sides' decisions in one request (the server batches them with every other game's)."""
    started = time.perf_counter()
    prepared = [(side, runner, observation, runner.session.prepare(observation)) for side, runner, observation in ready]
    client = ready[0][1].session.client
    results = client.decide_many([(runner.session.sid, payload)
                                  for _side, runner, _obs, (_host, _stored, payload) in prepared])
    elapsed = (time.perf_counter() - started) * 1000
    decided = {}
    for (side, runner, observation, (host, stored, _payload)), (actions, log_prob, value) in zip(prepared, results):
        decision = runner.session.finish(observation, host, stored, actions, log_prob, value, elapsed)
        decided[side] = runner.moves(decision)
    return decided


def main(argv: list[str]) -> int:
    import torch
    import firstlight_bot  # noqa: F401  (puts FirstLight's native_runner on the path)
    from il.speed import install
    install()
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, help='the RL run folder (its latest.pt is the model `latest`)')
    parser.add_argument('--address', type=_address, default=DEFAULT_ADDRESS)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-batch', type=int, default=GRAPH_BUCKETS[-1])
    parser.add_argument('--eager', action='store_true', help='no CUDA graphs (every forward launched op by op)')
    args = parser.parse_args(argv)
    Server(args.run, torch.device(args.device), args.max_batch, graphs=not args.eager).serve(args.address)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
