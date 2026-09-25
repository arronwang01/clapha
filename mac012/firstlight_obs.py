"""Build FirstLight's ObservationV1 from our live frames, without their injected probe.

Why this is possible at all: FirstLight's V4 tensorizer accepts FAIR actor observations only
(tensorizer.py rejects ORACLE outright), so it never expects the opponent's hand -- which is
exactly the information this client withholds. The episode tensorizer can be built from a
hand-made EpisodeConfigV1, so their BattleEnv (and therefore their in-process probe) is not
needed to run a policy.

What we can supply in full: tick, entities, towers, both players' elixir, our own hand /
deck / cycle, and both players' revealed cards.
What we cannot: the fine-grained combat event stream (damage attribution, shields, effects),
which their probe collects from inside the process. We emit the events we can derive from
frame diffs and leave the rest out.

Every field here is either read or derived from a read. Where FirstLight has a constant or a
table for something (the match timeline, the princess tower troop id, the velocity unit), this
module imports or reproduces *theirs* rather than inventing one, so the numbers the policy
sees are in the frame it was trained in.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FIRSTLIGHT = Path(os.environ.get('FIRSTLIGHT_ROOT')
                  or Path.home() / 'Documents/GitHub/FirstLight_CR')
if str(FIRSTLIGHT) not in sys.path:
    sys.path.insert(0, str(FIRSTLIGHT))

CLAPHA = Path(__file__).resolve().parents[1]
CARDS = {c['card_id']: c for c in
         json.loads((CLAPHA / 'live_card_catalog.json').read_text())['cards']}

# FirstLight distinguishes left and right side towers: TOWER_TYPE = king / princess_left /
# princess_right. Left and right are in native x (3500 = left, 14500 = right).
TOWERS = ((9000, 3000, 0, 'king'),
          (3500, 6500, 0, 'princess_left'), (14500, 6500, 0, 'princess_right'),
          (9000, 29000, 1, 'king'),
          (3500, 25500, 1, 'princess_left'), (14500, 25500, 1, 'princess_right'))

# FirstLight requires an exact tower troop identity for side towers. This is their own
# constant (match_factory.PRINCESS_TOWER_TROOP_ID); which troop is *equipped* is not
# something our reader exposes, so the plain Tower Princess is an assumption.
TOWER_PRINCESS = 159_000_000

TICK_MS = 50

_TIMELINE = None


def timeline():
    """FirstLight's own standard-mode timeline, loaded from their shipped CSV assets.

    The policy's clock features come from this: remaining time, phase and the elixir
    multiplier. Leaving them unset made every frame look like a match with zero time left
    running at single elixir, which is not a neutral default -- it is a wrong reading.
    """
    global _TIMELINE
    if _TIMELINE is None:
        from native_runner.battle_env import STANDARD_GAME_MODE
        from native_runner.timeline import load_game_mode_timeline
        _TIMELINE = load_game_mode_timeline(STANDARD_GAME_MODE).timeline
    return _TIMELINE


def gameplay_end_tick() -> int:
    from native_runner.match_factory import NATIVE_GAMEPLAY_END_TICK
    return int(NATIVE_GAMEPLAY_END_TICK)


_GLOBAL_BY_CARD: dict[int, int] | None = None


def global_id_by_card() -> dict[int, int]:
    """card id -> native_data_global_id, built from FirstLight's OWN shipped catalogs.

    The tensorizer resolves an entity's archetype from native_data_global_id, which an
    external reader cannot see. But the mapping is static, not match-specific, so it can be
    rebuilt offline:  card -> CardSpecV1.summoned_forms name -> form_vocab_id -> invert the
    catalog's global-id->vocab table -> global id.
    """
    global _GLOBAL_BY_CARD
    if _GLOBAL_BY_CARD is None:
        from native_runner.training.v4.factory import production_semantic_bundle
        bundle = production_semantic_bundle()
        catalog = bundle.entity_archetype_catalog
        by_vocab: dict[int, int] = {}
        for global_id, vocab in catalog._runtime_vocab_by_global_id.items():
            by_vocab.setdefault(int(vocab), int(global_id))
        table: dict[int, int] = {}
        for spec in bundle.native_card_catalog.specs:
            for form in getattr(spec, 'summoned_forms', ()) or ():
                vocab = catalog.form_vocab_id(form)
                if vocab > 1 and vocab in by_vocab:
                    table[int(spec.card_id)] = by_vocab[vocab]
                    break
        _GLOBAL_BY_CARD = table
    return _GLOBAL_BY_CARD


_KNOWN_CARDS: frozenset[int] | None = None


def known_cards() -> frozenset[int]:
    """Cards FirstLight's own card catalog can name.

    The tensorizer raises on a publicly revealed card it cannot place
    ("global card catalog is missing public opponent cards"), and again if more than eight
    are revealed. Both are fatal to a decision, so revealed lists are filtered through this
    and capped before they ever reach it.
    """
    global _KNOWN_CARDS
    if _KNOWN_CARDS is None:
        from native_runner.training.v4.factory import production_semantic_bundle
        from native_runner.training.v4.tensorizer import UNKNOWN_CARD_VOCAB_ID
        bundle = production_semantic_bundle()
        catalog = bundle.card_catalog
        _KNOWN_CARDS = frozenset(card_id for card_id in bundle.card_specs
                                 if catalog.vocab_id(card_id) != UNKNOWN_CARD_VOCAB_ID)
    return _KNOWN_CARDS


def public_cards(cards) -> tuple[int, ...]:
    """The first eight distinct revealed cards their catalog recognises, in reveal order."""
    known = known_cards()
    out: list[int] = []
    for card_id in cards or ():
        value = int(card_id)
        if value in known and value not in out:
            out.append(value)
        if len(out) == 8:
            break
    return tuple(out)


_ARCHETYPE: dict[int, tuple[int, str]] | None = None


def archetype_by_card() -> dict[int, tuple[int, str]]:
    """card id -> (native_data_global_id, entity kind), both from FirstLight's own catalog.

    The kind has to come from the catalog rather than from the card's type, because their
    child-kind vocabulary is character / building / projectile / area and has no "spell" in
    it. A card whose board entity cannot be resolved here is absent from this table, and is
    left out of the observation rather than sent in as an unknown -- the V4 tensorizer runs
    with reject_unknown_public_semantics=True, so an unknown entity raises and costs the bot
    every decision for as long as that entity is alive.
    """
    global _ARCHETYPE
    if _ARCHETYPE is None:
        from native_runner.training.v4.factory import production_semantic_bundle
        catalog = production_semantic_bundle().entity_archetype_catalog
        table: dict[int, tuple[int, str]] = {}
        for card_id, global_id in global_id_by_card().items():
            vocab = catalog.runtime_global_vocab_id(global_id)
            metadata = catalog.metadata_for_vocab_id(vocab)
            kind = _ENTITY_KIND.get(str(metadata.child_kind))
            if metadata.child_kind_known and kind is not None:
                table[int(card_id)] = (int(global_id), kind)
        # Hero and evolution units carry their FORM id (hero_form_id 203xxxxxx, evolution
        # 13xxxxxx) where a normal unit carries its card id, so a hero Musketeer arrived as
        # 203000014 and was dropped from the model's view entirely. Their catalog names those
        # forms (form:MusketeerHero, form:Musketeer_EV1); resolve them the same way.
        by_vocab: dict[int, int] = {}
        for global_id, vocab in catalog._runtime_vocab_by_global_id.items():
            by_vocab.setdefault(int(vocab), int(global_id))
        for info in CARDS.values():
            for form_id_key, form_name_key in (('hero_form_id', 'hero_character'),
                                               ('evolution_form_id', 'evolution_form')):
                form_id, form_name = info.get(form_id_key), info.get(form_name_key)
                if not form_id or not form_name or int(form_id) in table:
                    continue
                vocab = catalog.form_vocab_id(form_name)
                if vocab <= 1 or vocab not in by_vocab:
                    continue
                metadata = catalog.metadata_for_vocab_id(vocab)
                kind = _ENTITY_KIND.get(str(metadata.child_kind))
                if metadata.child_kind_known and kind is not None:
                    table[int(form_id)] = (by_vocab[vocab], kind)
        _ARCHETYPE = table
    return _ARCHETYPE


# The catalog's child kinds, in the vocabulary BattleEnv itself emits. A catalog
# "character" must be handed over as "troop": the tensorizer keeps our word for a character
# (_entity_child_kind), so saying "character" would file every troop under a different child
# type than the one these checkpoints were trained on.
_ENTITY_KIND = {'character': 'troop', 'building': 'building',
                'projectile': 'projectile', 'area': 'area'}


_RULESET_ID: str | None = None


def ruleset_id() -> str:
    """The ruleset identity their PPO league runs used (ppo_ruleset_id_v4).

    It is provenance, not geometry -- the tensorizer never reads it -- but card_placement_mask
    requires a SHA-256 digest, and the id the checkpoints were trained under is the honest one.
    """
    global _RULESET_ID
    if _RULESET_ID is None:
        from native_runner.training.v4.ppo_runtime import (ppo_environment_config_v4,
                                                           ppo_ruleset_id_v4)
        _RULESET_ID = ppo_ruleset_id_v4(ppo_environment_config_v4())
    return _RULESET_ID


class Battle:
    """Per-battle derived state: stable entity ids, ages, velocities, tower history.

    A frame on its own cannot say how fast anything is moving, how long it has been alive or
    whether a tower that is absent was destroyed or never seen. Those need memory, so one of
    these lives for the length of a battle and is rebuilt when the battle changes.
    """

    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self.start_tick: int | None = None
        self.ids: dict[str, int] = {}
        self.born: dict[str, int] = {}
        self.previous: dict[str, tuple[int, int, int]] = {}
        self.tower_max: dict[int, float] = {}
        self.tower_seen: set[int] = set()
        self.unresolved: dict[int, int] = {}
        self.untracked_plays: set[tuple[int, int]] = set()

    def tick(self, raw_tick: int) -> int:
        """The battle clock, already on FirstLight's own 0..6000 timeline.

        game_tick is battle-relative here, not an absolute counter: a draw freezes it at
        ~6150, just past their NATIVE_GAMEPLAY_END_TICK of 6000, and ~1080 read as 54 s at
        20 Hz. So no episode-start offset is subtracted -- doing that would restart the clock
        at zero whenever the bot is switched on partway through a battle.
        """
        if self.start_tick is None:
            self.start_tick = int(raw_tick)
        return max(0, int(raw_tick))

    def identify(self, address: str, tick: int) -> tuple[int, int, int]:
        """-> (stable entity id, age in ms, birth tick). Keyed on the heap address."""
        entity_id = self.ids.get(address)
        if entity_id is None:
            entity_id = self.ids[address] = len(self.ids) + 1
            self.born[address] = tick
        birth = self.born.get(address, tick)
        return entity_id, max(0, tick - birth) * TICK_MS, birth

    def velocity(self, address: str, tick: int, x: int, y: int):
        """Native units per tick, which is the unit BattleEnv derives velocity in."""
        previous = self.previous.get(address)
        self.previous[address] = (tick, x, y)
        if previous is None:
            return None
        last_tick, last_x, last_y = previous
        gap = tick - last_tick
        if gap <= 0:
            return None
        return ((x - last_x) / gap, (y - last_y) / gap)

    def forget(self, live_addresses: set[str]) -> None:
        """Drop entities that are gone, so a reused heap address cannot inherit a velocity."""
        for address in [key for key in self.previous if key not in live_addresses]:
            del self.previous[address]


_PLACEMENT_CACHE: dict = {}
MIRROR_CARD_ID = 28000006


def _placement_entry(card_id: int, owner: int, lanes, towers, buildings, unknown):
    """One placement entry, built exactly as BattleEnvV1._cached_placement_entry builds it.

    card_placement_mask is their own geometry: live tower rectangles, the river and bridges,
    destroyed-tower lane expansion, live building rectangles, and each building's native anchor
    lattice. It replaces the half-board approximation this module used to send, which offered
    tiles under our own towers as legal and gave a building no anchor at all -- the latter is
    what raised "building candidate and native placement anchor disagree" on every Cannon turn.
    Entries are cached on the same key so identical inputs return the identical object, which
    the tensorizer's own placement cache relies on.
    """
    from native_runner.arena import card_placement_mask, uses_entity_deployment_center
    from native_runner.building_placement_profiles import building_placement_profile
    from native_runner.contracts import content_hash
    from native_runner.training.v4.factory import production_semantic_bundle

    spec = production_semantic_bundle().card_specs[card_id]
    profile = building_placement_profile(card_id)
    stationary = bool(spec.kind.value == 'building' and profile is not None
                      and profile.stationary_collision_rectangle)
    tower_sig = (tuple((t.center_x_units, t.center_y_units, t.width_tiles, t.height_tiles)
                       for t in towers) if uses_entity_deployment_center(spec) else ())
    occupied_sig = (tuple((b.center_x_units, b.center_y_units, b.width_tiles, b.height_tiles)
                          for b in buildings) if stationary else ())
    unknown_sig = tuple(sorted(set(unknown))) if stationary else ()
    key = (card_id, owner, 'base', lanes, tower_sig, occupied_sig, unknown_sig)
    cached = _PLACEMENT_CACHE.get(key)
    if cached is not None:
        return cached
    artifact = card_placement_mask(spec, owner, ruleset_id=ruleset_id(), form='base',
                                   destroyed_enemy_princess_lanes=lanes,
                                   active_tower_footprints=towers,
                                   occupied_building_footprints=buildings)
    rows, accuracy, reasons, mask_id = (artifact.rows, artifact.accuracy.value,
                                        artifact.reasons, artifact.mask_id)
    if stationary and unknown_sig:
        # Their env refuses to place a building next to a live building whose footprint it
        # cannot verify, rather than guess at the overlap. Same here.
        rows = tuple(tuple(False for _ in range(18)) for _ in range(32))
        accuracy = 'blocked'
        reasons = tuple(sorted({*artifact.reasons, 'unverified_live_building_obstacle:'
                                + ','.join(str(v) for v in unknown_sig)}))
        mask_id = content_hash({'base_mask_id': artifact.mask_id, 'rows': rows,
                                'accuracy': accuracy, 'reasons': reasons})
    entry = {'card_id': card_id, 'shape': (32, 18), 'row_major': rows,
             'placement_rule': artifact.rule.value, 'accuracy': accuracy, 'mask_id': mask_id,
             'algorithm': artifact.algorithm,
             'collision_radius_units': artifact.collision_radius_units,
             'footprint_width_tiles': artifact.footprint_width_tiles,
             'footprint_height_tiles': artifact.footprint_height_tiles,
             'model_subcell_offset': artifact.model_subcell_offset,
             'form': artifact.form, 'reasons': reasons}
    if len(_PLACEMENT_CACHE) >= 4096:
        _PLACEMENT_CACHE.clear()
    _PLACEMENT_CACHE[key] = entry
    return entry


def placement_context(tower_states, frame: dict, owner: int):
    """The three inputs their env derives before building masks: destroyed enemy princess
    lanes, active tower rectangles, and live building rectangles (plus any live building whose
    footprint is unverified)."""
    from native_runner.arena import OccupiedFootprintV1, native_building_footprint
    from native_runner.building_placement_profiles import building_placement_profile
    from native_runner.training.v4.factory import production_semantic_bundle

    lanes = tuple(sorted({'left' if t.tower_kind == 'princess_left' else 'right'
                          for t in tower_states
                          if t.owner == 1 - owner and not t.active
                          and t.tower_kind in ('princess_left', 'princess_right')}))
    towers = tuple(OccupiedFootprintV1(float(t.position[0]), float(t.position[1]),
                                       4 if t.tower_kind == 'king' else 3,
                                       4 if t.tower_kind == 'king' else 3,
                                       f'tower:{t.entity_id}:{t.tower_kind}')
                   for t in tower_states if t.active)
    specs = production_semantic_bundle().card_specs
    occupied: dict = {}
    unknown: set[int] = set()
    for e in frame['entities']:
        if e['card_id'] == -1 or (e.get('hp') or 0) <= 0:
            continue
        spec = specs.get(e['card_id'])
        if spec is None or spec.kind.value != 'building':
            continue
        profile = building_placement_profile(e['card_id'])
        if profile is not None and profile.deploy_target_ready \
                and not profile.stationary_collision_rectangle:
            continue
        footprint = native_building_footprint(spec)
        if footprint is None:
            unknown.add(int(e['card_id']))
            continue
        signature = (e['x'], e['y'], footprint.width_tiles, footprint.height_tiles)
        occupied[signature] = OccupiedFootprintV1(float(e['x']), float(e['y']),
                                                  footprint.width_tiles, footprint.height_tiles,
                                                  f"building:{e['card_id']}:{e.get('address')}")
    return lanes, towers, tuple(occupied[k] for k in sorted(occupied)), tuple(sorted(unknown))


def action_mask(frame: dict, side: int, elixir: float, reserved: float = 0.0,
                tower_states=(), hand_forms: dict | None = None):
    """Which hand slots are playable, and where -- built the way BattleEnvV1.action_mask does.

    reserved is elixir already committed to a play the server has not acknowledged yet. The
    client does not debit it for ~20 ticks, so without holding it back a second play can be
    chosen against elixir that is already spent. reasons['reserved_elixir'] is their own field
    for this and is read straight into the scalar features.

    hand_forms maps card id -> the form it would be played in (0 normal, 1 evolution, 2 hero).
    The entry's form_code must agree with hand_runtime_by_slot, or the tensorizer raises.
    """
    from native_runner.contracts import ActionKind, ActionMaskV1
    from native_runner.training.v4.factory import production_semantic_bundle

    specs = production_semantic_bundle().card_specs
    available = max(0.0, elixir - max(0.0, reserved))
    lanes, towers, buildings, unknown = placement_context(tower_states, frame, side)
    player = next((p for p in frame['players'] if p['side'] == side), None)
    slots: list[bool] = [False, False, False, False]
    masks: dict[str, dict] = {}
    reasons: dict[str, object] = {}
    if player:
        deck = player.get('deck_card_ids') or []
        for position, index in enumerate(player.get('hand_deck_indices') or []):
            if position > 3 or not (deck and 0 <= index < len(deck)):
                continue
            card_id = int(deck[index])
            spec = specs.get(card_id)
            if spec is None or spec.elixir_cost is None:
                reasons[str(position)] = 'card_spec_unavailable'
                continue
            if card_id == MIRROR_CARD_ID:
                # Mirror plays whatever was last played, and their decoder needs that card's
                # runtime contract, which we do not assemble yet. Offered as illegal rather
                # than crashing the turn; the only card in the 122-card sweep that failed.
                reasons[str(position)] = 'mirror_unsupported'
                continue
            cost = float(spec.elixir_cost)
            form = int((hand_forms or {}).get(card_id, 0))
            base = _placement_entry(card_id, side, lanes, towers, buildings, unknown)
            entry = {**base, 'visible_card_id': card_id, 'effective_card_id': card_id,
                     'native_effective_card_id': card_id, 'effective_cost': cost,
                     'form_code': form, 'native_form_code': form,
                     'source_native_hand_slot': position}
            masks[str(position)] = entry
            if cost > available:
                reasons[str(position)] = 'insufficient_elixir'
            elif not any(any(row) for row in entry['row_major']):
                reasons[str(position)] = 'no_legal_placement'
            else:
                reasons[str(position)] = 'legal'
                slots[position] = True
    kinds = {ActionKind.WAIT.value: True, ActionKind.PLAY_CARD.value: any(slots)}
    return ActionMaskV1(kinds=kinds, hand_slots=tuple(slots), placement_masks=masks,
                        reasons={**reasons, 'effective_elixir': available,
                                 'reserved_elixir': max(0.0, reserved)})


_UNTAG = 0x00FFFFFFFFFFFFFF
_BASE_CARD: dict[int, int] | None = None


def base_card(card_id: int) -> int:
    """A hero or evolution form id -> its base card id; a card id -> itself."""
    global _BASE_CARD
    if _BASE_CARD is None:
        table = {}
        for info in CARDS.values():
            for key in ('hero_form_id', 'evolution_form_id'):
                if info.get(key):
                    table[int(info[key])] = int(info['card_id'])
        _BASE_CARD = table
    return _BASE_CARD.get(int(card_id), int(card_id))


def _hit_speed(card_id: int) -> int | None:
    from native_runner.training.v4.factory import production_semantic_bundle
    spec = production_semantic_bundle().card_specs.get(base_card(card_id))
    value = getattr(spec, 'hit_speed_ms', None) if spec is not None else None
    return int(value) if value else None


def troop_runtime(e: dict, entity_id: int, tick: int, target_id: int | None,
                  target_known: bool, hit_speed: int | None):
    """(attack_state, movement_runtime, deployment_runtime) from our raw component reads,
    produced by FirstLight's OWN resolvers in native_runner.phase_runtime.

    Raw inputs (all read from the troop object; offsets per their probe layout headers, the
    component vtables located on this build with src/comp_probe.c):
      attack component  +0x10 target, +0x20 sequence stage, +0x24 timeline ms, +0x28 load ms
      movement component +0x1e0 classic-charge progress (-1 = unavailable)
      troop +0x15c deploy remaining ms

    What their resolver cannot give without their in-process hooks is the attack START edge,
    which is how they label WINDUP. The snapshot does show it, though: a target, no load left
    and the timeline running is exactly an attack waiting to release, and the time to release
    is their own formula (hit_speed - timeline mod hit_speed, attack dash 0). That one label is
    inferred here and marked so in the provenance; everything else is their resolver's output.
    RELEASE and INTERRUPTED are single-tick hook edges and stay unknown.
    """
    from dataclasses import replace
    from native_runner import phase_runtime as P
    from native_runner.contracts import AttackPhase

    key = (int(entity_id), 0, 0)
    attack = movement = deployment = None
    if e.get('has_attack'):
        timeline = max(0, int(e.get('atk_timeline', 0)))
        load = max(0, int(e.get('atk_load', 0)))
        stage = int(e.get('atk_stage', -1))
        raw = P.AttackRuntimeRaw(
            entity_key=key, tick=tick, target_validated=target_known,
            target_entity=target_id, attack_sequence_stage=stage if 0 <= stage <= 255 else None,
            attack_timeline_ms=timeline, load_remaining_ms=load,
            hit_speed_ms=hit_speed, attack_dash_time_ms=0 if hit_speed else None,
            attack_step_native_ms=P.SIMULATION_TICK_MS, hook_set_attested=True)
        attack = P.resolve_attack_runtime(raw).state
        if (attack.phase == AttackPhase.UNKNOWN and target_id is not None and load == 0
                and timeline > 0 and hit_speed):
            provenance = attack.provenance
            notes = tuple(getattr(provenance, 'notes', ()) or ()) + (
                'windup inferred from snapshot (target set, load 0, timeline running); '
                'no attack-start hook on the external reader',)
            attack = replace(attack, phase=AttackPhase.WINDUP,
                             phase_remaining_ms=hit_speed - timeline % hit_speed,
                             provenance=replace(provenance, notes=notes))
    if e.get('has_move'):
        charge = int(e.get('charge', -1))
        movement = P.resolve_movement_runtime(P.MovementRuntimeRaw(
            entity_key=key, tick=tick, component_validated=True, hook_set_attested=False,
            movement_delta=None, effective_speed=None, speed_input=None,
            speed_after_effects=None, classic_charge_progress=charge if charge >= -1 else None,
            classic_charge_speed_multiplier=None)).to_mapping()
    remaining = e.get('deploy_remaining')
    if remaining is not None and int(remaining) >= 0:
        deployment = P.resolve_deployment_runtime(P.DeploymentRuntimeRaw(
            tick=tick, component_validated=True, remaining_native_ms=int(remaining),
            previous_remaining_native_ms=None, configured_deploy_time_ms=None,
            uses_effect_scaled_step=None, observed_step_native_ms=None)).to_mapping()
    return attack, movement, deployment


def runtime_provenance(fields, attack, movement, deployment, tick: int):
    """Per-domain evidence for a troop or tower: which runtime domains were really observed.

    Their contract refuses a domain carrying data without positive provenance, and just as
    firmly refuses one that claims data it does not carry -- so each domain is declared
    exactly when we filled it, as NATIVE_AUTHORITATIVE (read straight from the object's
    component), and every other domain stays UNKNOWN.
    """
    from native_runner.contracts import SemanticEvidenceLevel, SemanticProvenanceV1
    base = SemanticProvenanceV1.unknown_all(fields)
    evidence = dict(base.field_evidence)
    sources: dict[str, tuple[str, ...]] = {}
    for name, value, origin in (
            ('attack_state', attack, ('attack component +0x10/+0x20/+0x24/+0x28',)),
            ('movement_runtime', movement, ('movement component +0x1e0',)),
            ('deployment_runtime', deployment, ('object +0x15c',))):
        if value is not None and name in evidence:
            evidence[name] = SemanticEvidenceLevel.NATIVE_AUTHORITATIVE
            sources[name] = origin
    return SemanticProvenanceV1(field_evidence=evidence, source_fields=sources,
                                observed_tick=tick,
                                notes=('external read-only reader; component layout per '
                                       'FirstLight probe headers',))


EVENT_WINDOW_TICKS = 45 * 20   # battle_env.EVENT_WINDOW_TICKS


def play_events(plays, tick: int, decks: dict | None, battle):
    """action_executed events for card plays that executed within their 45 s window.

    This is the one event FirstLight's public tracker learns from
    (DeterministicPublicTracker.CARD_EXECUTION_EVENTS). Without it the tracker never sees a
    play: the opponent's elixir estimate only ever climbs to full, and their cycle and card
    availability stay unknown for the whole match. Built from the game's own command queue
    (card, owner, position, and the tick it was consumed), shaped like battle_env's.

    The tracker refuses a play of a card outside that player's episode deck, so a play is
    only emitted when its card is in `decks[side]`; anything else is counted on the battle
    as unplaceable rather than raising and losing the decision.
    """
    from native_runner.contracts import EventV1
    from native_runner.training.v4.factory import production_semantic_bundle

    specs = production_semantic_bundle().card_specs
    events = []
    seen_commands = set()
    for play in plays or ():
        if not tick - EVENT_WINDOW_TICKS <= play['tick'] <= tick:
            continue
        # The same command can be reported twice (seen missing early, then dated from its
        # issue tick). It is one play.
        command = (play.get('side'), play.get('issue_tick'), play.get('seq'))
        if play.get('issue_tick') is not None:
            if command in seen_commands:
                continue
            seen_commands.add(command)
        if play.get('kind', 'card') != 'card':
            continue   # champion ability activations: not a card play (see viewer)
        card_id, side = int(play['card_id']), int(play['side'])
        if card_id == MIRROR_CARD_ID or (decks and card_id not in decks.get(side, ())):
            battle.untracked_plays.add((side, card_id))
            continue
        spec = specs.get(card_id)
        if spec is None or spec.elixir_cost is None:
            battle.untracked_plays.add((side, card_id))
            continue
        position = None
        if play.get('x') is not None and play.get('y') is not None:
            position = (float(play['x']), float(play['y']))
        events.append(EventV1(
            tick=int(play['tick']), event_type='action_executed', owner=side, card_id=card_id,
            position=position,
            data={'native_sequence': play.get('seq'),
                  'native_event_id': f"{play.get('issue_tick')}:{play.get('seq')}:{side}",
                  'visible_card_id': card_id, 'effective_card_id': card_id,
                  'native_effective_card_id': card_id,
                  'effective_cost': float(spec.elixir_cost), 'form_code': 0,
                  'native_form_code': 0, 'source': 'native_command_queue'}))
    events.sort(key=lambda event: (event.tick, event.owner or 0))
    return tuple(events)


def build(frame: dict, health: dict, episode_id: str, deduced: dict | None = None,
          revealed: dict | None = None, battle: Battle | None = None,
          reserved: float = 0.0, plays=None, decks: dict | None = None,
          hand_forms: dict | None = None):
    """One ObservationV1 for the local actor, FAIR tier.

    Pass the same Battle for every frame of a battle: velocity, entity identity, age, the
    clock and the crown count all need continuity, and without it the policy sees a board
    where nothing moves, nothing has history and no tower has ever fallen.

    hand_forms: card id -> form for our hand (see action_mask).
    """
    from native_runner.contracts import (ENTITY_RUNTIME_SEMANTIC_FIELDS,
                                         TOWER_RUNTIME_SEMANTIC_FIELDS,
                                         CausalGroupKind, CausalGroupRefV1, EntityStateV1,
                                         ObservationTier, ObservationV1, PlayerStateV1,
                                         TimeStateV1, TowerStateV1)

    revealed = revealed or {}
    side = health.get('local_side')
    if battle is None or battle.episode_id != episode_id:
        battle = Battle(episode_id)
    tick = battle.tick(frame['game_tick'])
    live = {(e['x'], e['y']): e for e in frame['entities'] if e['card_id'] == -1}

    phase, multiplier = timeline().phase(tick)
    remaining = max(0, gameplay_end_tick() - tick) * TICK_MS

    towers = []
    tower_rows: list = []
    id_by_address: dict[int, int] = {}
    crowns = {0: 0, 1: 0}
    tower_ids: dict[int, list[int]] = {0: [], 1: []}
    for index, (x, y, owner, kind) in enumerate(TOWERS):
        found = live.get((x, y))
        entity_id = 5000000 + index
        tower_ids[owner].append(entity_id)
        if found:
            battle.tower_seen.add(index)
            battle.tower_max[index] = float(found.get('max_hp') or 1)
            hitpoints = float(found.get('hp', 0))
        else:
            # Absent means destroyed only if we have seen it alive in this battle; before
            # that it is simply a tower we have not resolved, and must not score a crown.
            hitpoints = 0.0 if index in battle.tower_seen else float(battle.tower_max.get(index, 1))
        maximum = battle.tower_max.get(index, 1.0) or 1.0
        active = hitpoints > 0
        if not active and index in battle.tower_seen:
            crowns[1 - owner] += 1
        tower_rows.append((found, dict(
            entity_id=entity_id, owner=owner, tower_kind=kind,
            position=(float(x), float(y)),
            tower_troop_id=None if kind == 'king' else TOWER_PRINCESS,
            hitpoints=hitpoints, max_hitpoints=maximum, active=active)))
        if found and found.get('address'):
            id_by_address[int(found['address'], 16) & _UNTAG] = entity_id

    archetypes = archetype_by_card()
    entities = []
    addresses: set[str] = set()
    for e in frame['entities']:
        if e['card_id'] != -1 and archetypes.get(e['card_id']) is not None and e.get('address'):
            address = str(e['address'])
            known_id, _age, _birth = battle.identify(address, tick)
            id_by_address[int(address, 16) & _UNTAG] = known_id

    def target_of(e: dict) -> tuple[int | None, bool]:
        raw_target = int(str(e.get('target') or '0x0'), 16) & _UNTAG
        if raw_target == 0:
            return None, True
        found_id = id_by_address.get(raw_target)
        return found_id, found_id is not None

    for e in frame['entities']:
        if e['card_id'] == -1:
            continue
        resolved = archetypes.get(e['card_id'])
        if resolved is None:
            # Spell area effects are the usual case: our reader reports the spell's card id,
            # which has no entity archetype, and a strict tensorizer raises on it. Leaving the
            # entity out loses a short-lived effect; sending it in loses every decision while
            # it is on the board.
            battle.unresolved[e['card_id']] = battle.unresolved.get(e['card_id'], 0) + 1
            continue
        global_id, kind = resolved
        address = str(e.get('address') or f"{e['x']}:{e['y']}:{e['card_id']}")
        addresses.add(address)
        entity_id, age_ms, birth = battle.identify(address, tick)
        id_by_address[int(address, 16) & _UNTAG if address.startswith('0x') else -1] = entity_id
        target_id, target_known = target_of(e)
        attack, movement, deployment = troop_runtime(e, entity_id, tick, target_id,
                                                     target_known, _hit_speed(e['card_id']))
        # One deployment is one causal group. Without this every unit is a singleton
        # (grouping.causal_group_key falls back to "singleton:<id>"), so a Skeletons or
        # Minions play reads as three or four unrelated individuals rather than the swarm the
        # policy was trained on. The spawn cohort -- same owner, same card, same birth tick --
        # is exactly a deployment, and is the most we can honestly assert from frame diffs.
        group = CausalGroupRefV1(
            kind=CausalGroupKind.PERSISTENT_EFFECT if kind in ('area', 'effect')
            else (CausalGroupKind.VOLLEY if kind == 'projectile'
                  else CausalGroupKind.DEPLOYMENT),
            handle=f"{e['side']}:{e['card_id']}:{birth}",
            source_card_id=int(e['card_id']))
        entities.append(EntityStateV1(
            native_data_global_id=global_id,
            entity_id=entity_id, owner=e['side'], card_id=e['card_id'],
            entity_kind=kind,
            position=(float(e['x']), float(e['y'])),
            velocity=battle.velocity(address, tick, e['x'], e['y']),
            hitpoints=float(e['hp']), max_hitpoints=float(e['max_hp'] or 1),
            age_ms=age_ms, visible=True, causal_group=group,
            visible_target=target_id, attack_state=attack,
            movement_runtime=movement, deployment_runtime=deployment,
            runtime_provenance=runtime_provenance(ENTITY_RUNTIME_SEMANTIC_FIELDS, attack,
                                                  movement, deployment, tick)))
    battle.forget(addresses)

    for found, fields in tower_rows:
        attack = None
        target_id = None
        if found and found.get('has_attack'):
            target_id, target_known = target_of(found)
            attack, _movement, _deployment = troop_runtime(
                found, fields['entity_id'], tick, target_id, target_known, None)
        towers.append(TowerStateV1(
            **fields, visible_target=target_id, attack_state=attack,
            runtime_provenance=runtime_provenance(TOWER_RUNTIME_SEMANTIC_FIELDS, attack,
                                                  None, None, tick)))

    own_elixir = next((p['elixir_raw'] / 10000.0 for p in frame['players']
                       if p['side'] == side), 0.0)
    players = []
    for p in frame['players']:
        deck = p.get('deck_card_ids') or []
        # Only the actor's own hand is private state it may see. After settlement this
        # client exposes both hands; the opponent's must still go in as public-only.
        readable = p['side'] == side and p['hand_deck_indices'][0] != -1
        hand_slots = p['hand_deck_indices'] if readable else (
            (deduced or {}).get(p['side'], ([], None))[0])
        hand = tuple(deck[i] for i in hand_slots if deck and 0 <= i < len(deck))
        cycle = tuple(deck[i] for i in (p.get('cycle_deck_indices') or [])
                      if deck and 0 <= i < len(deck))
        nxt = None
        if readable and deck and 0 <= p.get('next_deck_index', -1) < len(deck):
            nxt = deck[p['next_deck_index']]
        common = {'owner': p['side'], 'crowns': crowns[p['side']],
                  'tower_ids': tuple(tower_ids[p['side']]),
                  'revealed_cards': public_cards(revealed.get(p['side'], ()))}
        if readable:
            # The tensorizer needs an exact card -> hand-slot map and a per-slot runtime form
            # (FORM_NORMAL = 0; evolutions would carry their own code, which we do not read).
            slot_by_card = {str(deck[i]): pos for pos, i in enumerate(hand_slots)
                            if deck and 0 <= i < len(deck)}
            runtime_by_slot = {str(pos): {'form_code': int((hand_forms or {}).get(
                                   deck[i] if deck and 0 <= i < len(deck) else -1, 0))}
                               for pos, i in enumerate(hand_slots)}
            players.append(PlayerStateV1(
                elixir_exact=own_elixir,
                hand=hand, next_card=nxt, deck=tuple(deck), cycle=cycle,
                private_state_visible=True,
                metadata={'hand_slot_by_card': slot_by_card,
                          'hand_runtime_by_slot': runtime_by_slot},
                **common))
        else:
            # FAIR forbids handing the actor the opponent's exact private state, even though
            # this client's memory does expose their exact elixir. Only public facts go in:
            # the cards they have played. FirstLight is stricter than the environment.
            players.append(PlayerStateV1(private_state_visible=False, **common))

    observation = ObservationV1(
        tier=ObservationTier.FAIR, tick=tick, owner=side,
        time=TimeStateV1(elapsed_ms=tick * TICK_MS, remaining_ms=remaining,
                         server_time_ms=int(frame['game_tick']) * TICK_MS,
                         elixir_multiplier=multiplier, tick_ms=TICK_MS),
        phase=phase,
        players=tuple(players), towers=tuple(towers), entities=tuple(entities),
        events=play_events(plays, tick, decks, battle),
        action_mask=action_mask(frame, side, own_elixir, reserved, towers, hand_forms),
        episode_id=episode_id,
        ruleset_id=ruleset_id())
    return observation, battle
