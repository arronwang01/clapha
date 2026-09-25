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

# The up-to-date upstream clone kept in the project (update_check reports when it falls behind),
# else the user's own download.
_CLONE = Path(__file__).resolve().parents[1] / 'ref-firstlight'
FIRSTLIGHT = Path(os.environ.get('FIRSTLIGHT_ROOT')
                  or (_CLONE if (_CLONE / 'native_runner').is_dir()
                      else Path.home() / 'Documents/GitHub/FirstLight_CR'))
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


def regular_end_tick() -> int:
    """First overtime tick on the standard timeline (3600: three minutes at 20 Hz)."""
    end = gameplay_end_tick()
    low, high = 0, end
    while low < high:
        middle = (low + high) // 2
        if timeline().phase(middle)[0] == 'normal':
            low = middle + 1
        else:
            high = middle
    return low


def battle_result(battle: 'Battle', tick: int) -> tuple[int | None, str] | None:
    """(winning side or None for a draw, reason) once the battle is decided; else None.

    Standard 1v1 rules on the towers this battle has seen fall: a king tower ends it at any
    time; at full time (and on every tower after it -- sudden death) unequal crowns end it;
    at the end of overtime it goes to the tiebreak. The live client keeps the clock running
    for a few seconds after the end and accepts taps it will never execute, so the console
    stops acting on this rather than on the clock stopping.
    """
    if battle.start_tick is None or tick <= battle.start_tick:
        # A clock that has not moved since we started watching is a battle already over: its
        # frozen end state or its teardown, which frees tower objects one by one (a finished
        # 2026-09-25 win read as a loss that way). Its towers were never seen fall; no verdict.
        return None
    down = battle.towers_down
    kings = {TOWERS[index][2] for index in down if TOWERS[index][3] == 'king'}
    if kings:
        return (None, 'both king towers destroyed') if len(kings) == 2 else \
            (1 - kings.pop(), 'king tower destroyed')
    crowns = {0: 0, 1: 0}
    for index in down:
        crowns[1 - TOWERS[index][2]] += 1
    if tick >= regular_end_tick() and crowns[0] != crowns[1]:
        winner = 0 if crowns[0] > crowns[1] else 1
        last_fall = max(battle.tower_down_tick.get(index, 0) for index in down)
        when = 'in overtime' if last_fall >= regular_end_tick() else 'at full time'
        return winner, f'{crowns[winner]}-{crowns[1 - winner]} on crowns {when}'
    if tick >= gameplay_end_tick():
        return None, f'{crowns[0]}-{crowns[1]} at the end of overtime (tiebreak on tower health)'
    return None


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
        # The unit an evolution summons is named by FirstLight's own card spec
        # (evolution Transform effect 'summoned_form', e.g. Skeleton_EV1). Our catalog's
        # 'evolution_form' is the CARD-level name (Skeletons_EV1), which matches only where
        # card and unit share a name -- Evo Skeletons, Barbarians, Bats, Recruits, Royal Hogs
        # and Wall Breakers were all invisible to the model.
        specs = production_semantic_bundle().card_specs
        unit_form: dict[int, str] = {}
        for card_id, spec in specs.items():
            for effect in (spec.evolution.effects if spec.evolution is not None else ()):
                name = effect.parameters.get('summoned_form') if effect.parameters else None
                if name:
                    unit_form[int(card_id)] = str(name)
        for info in CARDS.values():
            for form_id_key, names in (
                    ('hero_form_id', (info.get('hero_character'),)),
                    ('evolution_form_id', (unit_form.get(int(info['card_id'])),
                                           info.get('evolution_form')))):
                form_id = info.get(form_id_key)
                if not form_id or int(form_id) in table:
                    continue
                for form_name in names:
                    if not form_name:
                        continue
                    vocab = catalog.form_vocab_id(form_name)
                    if vocab <= 1 or vocab not in by_vocab:
                        continue
                    metadata = catalog.metadata_for_vocab_id(vocab)
                    kind = _ENTITY_KIND.get(str(metadata.child_kind))
                    if metadata.child_kind_known and kind is not None:
                        table[int(form_id)] = (by_vocab[vocab], kind)
                        break
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
        self.towers_down: set[int] = set()      # TOWERS indices seen alive, now destroyed
        self.tower_hp: dict[int, float] = {}     # last readable health per tower
        self.tower_down_tick: dict[int, int] = {}  # when each destroyed tower was first seen down
        self.unreadable_hp = 0                   # unit readings left out: health read as < 0
        self.unresolved: dict[int, int] = {}
        self.form_fallbacks: set[int] = set()   # forms shown as their base unit
        self.untracked_plays: set[tuple[int, int]] = set()
        self.unresolved_abilities: set[int] = set()

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
        if e['card_id'] == -1 or is_effect(e) or (e.get('hp') or 0) <= 0:
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


# FirstLight's exact ability button enum (rich_telemetry_adapter.ABILITY_BUTTON_STATE_LABELS and
# _ability_phase). Every value the device showed fits it: 0 no match, 1 ChampionAbsent, 2 Ready,
# 6 AllChargesConsumed, 9 NotEnoughElixir. Only Ready and LimitedAvailability are queueable.
ABILITY_QUEUEABLE_BUTTON_STATES = frozenset((2, 4))
_ABILITY_PHASE_BY_BUTTON = {1: 'unavailable', 2: 'ready', 4: 'ready', 6: 'exhausted',
                            8: 'cooldown', 9: 'unavailable', 10: 'casting', 11: 'unavailable',
                            12: 'unavailable', 13: 'unavailable'}
_HERO_CARD_BY_CHARACTER: dict[int, int] | None = None
_ABILITY_BY_CARD: dict[int, tuple[str, object]] | None = None


def hero_card_by_character() -> dict[int, int]:
    """hero character data global id -> base card id.

    A controller names its hero by the character data it selected (+0x90 -> +0x40); the device
    showed 130283371 for Hero Musketeer and 2979504115 for Hero Ice Golem, which are exactly the
    archetype ids FirstLight's catalog gives those hero forms. Hero form -> base card is
    FirstLight's own HERO_FORM_TO_BASE_CARD.
    """
    global _HERO_CARD_BY_CHARACTER
    if _HERO_CARD_BY_CHARACTER is None:
        from native_runner.training.v4.native_actions import HERO_FORM_TO_BASE_CARD
        table: dict[int, int] = {}
        for form_id, (global_id, _kind) in archetype_by_card().items():
            base = HERO_FORM_TO_BASE_CARD.get(int(form_id))
            if base is None and 203000000 <= int(form_id) < 204000000:
                base = base_card(int(form_id))
            if base is not None:
                table[int(global_id) & 0xFFFFFFFF] = int(base)
        _HERO_CARD_BY_CHARACTER = table
    return _HERO_CARD_BY_CHARACTER


def ability_by_card() -> dict[int, tuple[str, object]]:
    """base card -> (ability id, AbilitySpec), only where the card has exactly one ability --
    the same unique catalog join their adapter requires before it emits an ability state."""
    global _ABILITY_BY_CARD
    if _ABILITY_BY_CARD is None:
        from native_runner.training.v4.factory import production_semantic_bundle
        seen: dict[int, list] = {}
        for ability_id, spec in production_semantic_bundle().ability_specs.items():
            card = getattr(spec, 'source_card_id', None)
            if card is not None:
                seen.setdefault(int(card), []).append((str(ability_id), spec))
        _ABILITY_BY_CARD = {card: rows[0] for card, rows in seen.items() if len(rows) == 1}
    return _ABILITY_BY_CARD


def _state_provenance(fields, filled: dict, tick: int, notes: tuple[str, ...]):
    from native_runner.contracts import SemanticEvidenceLevel, SemanticProvenanceV1
    evidence = dict(SemanticProvenanceV1.unknown_all(fields).field_evidence)
    sources = {}
    for name, (level, origin) in filled.items():
        evidence[name] = level
        sources[name] = origin
    return SemanticProvenanceV1(field_evidence=evidence, source_fields=sources,
                                observed_tick=tick, notes=notes)


def own_runtime_states(player: dict, entities, side: int, tick: int, battle,
                       evo_required: dict | None = None):
    """(ability_runtime_states, evolution_runtime_states) for the actor, from memory.

    Abilities: one per bound hero controller, joined controller -> selected hero character ->
    base card -> the card's single ability spec (id, elixir cost). The source entity is our
    live unit with that character's archetype. Phase and `available` follow their button enum.
    Evolutions: the player's per-deck-slot progress vector, for slots whose form flag has the
    evolution bit; ready = progress >= FirstLight's cycles required, exactly as their probe.
    """
    from native_runner.contracts import (ABILITY_RUNTIME_STATE_FIELDS,
                                         EVOLUTION_RUNTIME_STATE_FIELDS, AbilityPhase,
                                         AbilityRuntimeStateV1, EvolutionPhase,
                                         EvolutionRuntimeStateV1, SemanticEvidenceLevel)
    from native_runner.training.v4.factory import production_semantic_bundle

    native = SemanticEvidenceLevel.NATIVE_DERIVED
    static = SemanticEvidenceLevel.STATIC_DECLARED
    abilities = []
    heroes = hero_card_by_character()
    by_card = ability_by_card()
    for raw in player.get('abilities') or ():
        character = int(raw.get('character_id') or 0) & 0xFFFFFFFF
        if not character:
            continue
        card = heroes.get(character)
        joined = by_card.get(card) if card is not None else None
        if joined is None:
            battle.unresolved_abilities.add(character)
            continue
        ability_id, spec = joined
        sources = [e.entity_id for e in entities
                   if e.owner == side and (int(e.native_data_global_id or 0) & 0xFFFFFFFF) == character]
        source_entity = sources[0] if len(sources) == 1 else None
        button = int(raw.get('button', 0))
        phase = AbilityPhase(_ABILITY_PHASE_BY_BUTTON.get(button, 'unknown'))
        charges_raw = int(raw.get('charges', -1))
        cost = getattr(spec, 'elixir_cost', None)
        filled = {
            'phase': (native if phase != AbilityPhase.UNKNOWN else SemanticEvidenceLevel.UNKNOWN,
                      ('controller+0x98 button state',)),
            'available': (native, ('controller+0x98 button state',)),
            'cooldown_ms': (native, ('controller+0x7c',)),
            'remaining_cooldown_ms': (native, ('controller+0x78',)),
            'charges': (native, ('controller+0x80',)),
        }
        if cost is not None:
            filled['elixir_cost'] = (static, ('AbilitySpecV1.elixir_cost',))
        if source_entity is not None:
            filled['source_entity'] = (native, ('live unit with the controller character',))
        abilities.append(AbilityRuntimeStateV1(
            ability_id=ability_id, source_entity=source_entity, phase=phase,
            elixir_cost=float(cost) if cost is not None else None,
            cooldown_ms=int(raw.get('configured_ms', 0)),
            remaining_cooldown_ms=max(0, int(raw.get('cooldown_ms', 0))),
            charges=None if charges_raw == -1 else charges_raw,
            available=button in ABILITY_QUEUEABLE_BUTTON_STATES,
            attributes={'controller_slot': int(raw['controller_slot']),
                        'source_card_id': int(card), 'button_state': button,
                        'selected_character_data_global_id': character,
                        'remaining_charges_raw': charges_raw,
                        'classification': 'exact_catalog_runtime_join'},
            provenance=_state_provenance(ABILITY_RUNTIME_STATE_FIELDS, filled, tick,
                                         (f'raw ability enum={button}',))))

    evolutions = []
    deck = player.get('deck_card_ids') or []
    flags = player.get('deck_form_flags') or []
    progress = player.get('evo_progress') or []
    specs = production_semantic_bundle().card_specs
    if len(progress) == len(deck) == len(flags) == 8:
        for slot, (card, flag, value) in enumerate(zip(deck, flags, progress)):
            spec = specs.get(int(card))
            evolution = getattr(spec, 'evolution', None) if spec is not None else None
            if not int(flag or 0) & 0x1 or evolution is None or not evolution.cycle_required:
                continue
            required = int((evo_required or {}).get(int(card), evolution.cycle_required))
            value = max(0, int(value))
            ready = value >= required
            phase = (EvolutionPhase.READY if ready else
                     EvolutionPhase.BASE if value == 0 else EvolutionPhase.CYCLING)
            filled = {name: (native, ('player+0x2e8 progress vector',)) for name in
                      ('deck_slot', 'phase', 'cycle_required', 'cycle_remaining', 'ready',
                       'deployments_in_cycle')}
            filled['base_form_id'] = (static, ('CardSpecV1.evolution.base_form_id',))
            filled['next_form_id'] = (static, ('CardSpecV1.evolution.evolution_form_id',))
            evolutions.append(EvolutionRuntimeStateV1(
                card_id=int(card), deck_slot=slot, phase=phase,
                base_form_id=evolution.base_form_id, next_form_id=evolution.evolution_form_id,
                cycle_required=required, cycle_remaining=max(0, required - value),
                ready=ready, deployments_in_cycle=value,
                attributes={'raw_progress': value, 'classification': 'exact_catalog_runtime_join'},
                provenance=_state_provenance(EVOLUTION_RUNTIME_STATE_FIELDS, filled, tick, ())))
    return tuple(abilities), tuple(evolutions)


def legal_ability_sources(abilities, elixir: float, pending=()) -> tuple[int, ...]:
    """BattleEnvV1._ability_action_candidates' rule, verbatim in effect: Ready and available,
    no cooldown, charges left (or unlimited), an exact cost we can pay, a live source unit, and
    not already requested (a tap in flight)."""
    legal = []
    for state in abilities:
        if (state.source_entity is None or state.source_entity in pending
                or state.phase.value != 'ready' or state.available is not True
                or (state.remaining_cooldown_ms or 0) != 0
                or (state.charges is not None and state.charges <= 0)
                or state.elixir_cost is None or state.elixir_cost > elixir):
            continue
        legal.append(int(state.source_entity))
    return tuple(sorted(set(legal)))


def action_mask(frame: dict, side: int, elixir: float, reserved: float = 0.0,
                tower_states=(), hand_forms: dict | None = None, abilities=(),
                pending_ability_sources=()):
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
    ability_sources = legal_ability_sources(abilities, available, pending_ability_sources)
    kinds = {ActionKind.WAIT.value: True, ActionKind.PLAY_CARD.value: any(slots),
             ActionKind.ACTIVATE_ABILITY.value: bool(ability_sources)}
    return ActionMaskV1(kinds=kinds, hand_slots=tuple(slots), placement_masks=masks,
                        ability_sources=ability_sources,
                        reasons={**reasons, 'effective_elixir': available,
                                 'reserved_elixir': max(0.0, reserved)})


_UNTAG = 0x00FFFFFFFFFFFFFF
_ARCHETYPE_BY_DATA: dict[int, tuple[int, str] | None] = {}


def _addr(value) -> int | None:
    """A heap address as an untagged int; None for anything that is not one."""
    try:
        return int(str(value), 16) & _UNTAG
    except (TypeError, ValueError):
        return None


_CARD_BY_ARCHETYPE: dict[int, int] | None = None


def card_of_unit(global_id: int, fallback: int) -> int:
    """The card whose unit this archetype is (Barbarian -> Barbarians), for its stats; the
    object's own card id otherwise."""
    global _CARD_BY_ARCHETYPE
    if _CARD_BY_ARCHETYPE is None:
        _CARD_BY_ARCHETYPE = {}
        for card, (gid, _kind) in sorted(archetype_by_card().items()):
            _CARD_BY_ARCHETYPE.setdefault(int(gid), int(card))
    return _CARD_BY_ARCHETYPE.get(int(global_id), int(fallback))


def is_tower(e: dict) -> bool:
    """Towers are card -1 AND a tower object (kind 12/13). A tower's own shots are card -1 too,
    spawned at the tower's position; they are projectiles (kind 0), not towers."""
    return e.get('card_id') == -1 and e.get('kind', 12) != 0


def is_effect(e: dict) -> bool:
    """A projectile or area effect: kind 0, identified only by its own data record."""
    return e.get('kind') == 0


def archetype_by_data(data_id: int) -> tuple[int, str] | None:
    """An object's own data record -> (archetype global id, entity kind), from FirstLight's
    catalog. This is the key their environment uses (native_data_global_id): it names the unit a
    spawner produced (Battle Ram -> Barbarians, Tombstone -> Skeletons) and the ProjectileData /
    AreaEffectData of a spell or shot in flight, which the card id cannot."""
    data_id = int(data_id) & 0xFFFFFFFF
    if data_id not in _ARCHETYPE_BY_DATA:
        from native_runner.training.v4.factory import production_semantic_bundle
        catalog = production_semantic_bundle().entity_archetype_catalog
        result = None
        vocab = catalog.runtime_global_vocab_id(data_id) if data_id else 1
        if vocab > 1:
            metadata = catalog.metadata_for_vocab_id(vocab)
            kind = _ENTITY_KIND.get(str(metadata.child_kind))
            if metadata.child_kind_known and kind is not None:
                result = (data_id, kind)
        _ARCHETYPE_BY_DATA[data_id] = result
    return _ARCHETYPE_BY_DATA[data_id]
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


def projectile_state(e: dict, global_id: int, entity_id: int, velocity, tick: int):
    """FirstLight's ProjectileStateV1 for a shot or spell in flight, as their adapter builds it:
    in flight while the object exists, its velocity, its source card, its projectile data id.
    Damage and radius stay unset -- the tensorizer then takes the projectile archetype's catalog
    values, as it does for their probe. The destination is not read yet, so it stays unset too."""
    from native_runner.contracts import (PROJECTILE_STATE_FIELDS, ProjectilePhase,
                                         ProjectileStateV1, SemanticEvidenceLevel)
    derived = SemanticEvidenceLevel.NATIVE_DERIVED
    card = e['card_id'] if e.get('card_id', -1) > 0 else None
    filled = {'phase': (derived, ('object present in the battle entity list',))}
    if velocity is not None:
        filled['velocity'] = (derived, ('position delta / tick delta',))
    if card is not None:
        filled['source_card_id'] = (derived, ('object +0xac',))
    return ProjectileStateV1(
        projectile_id=f'projectile:{entity_id}:{global_id}', phase=ProjectilePhase.IN_FLIGHT,
        source_card_id=card, velocity=velocity,
        attributes={'native_projectile_data_global_id': int(global_id),
                    'terminal_reason': 'unknown'},
        provenance=_state_provenance(PROJECTILE_STATE_FIELDS, filled, tick,
                                     ('external read-only reader; destination not read',)))


def runtime_provenance(fields, attack, movement, deployment, tick: int, projectile=None):
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
            ('deployment_runtime', deployment, ('object +0x15c',)),
            ('projectile_state', projectile, ('object +0x48 ProjectileData, frame deltas',))):
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
    only emitted when its card is in `decks[side]`. The console registers every card a player
    shows before calling this (FirstLightRunner.register_plays), so `decks` holds what the
    tracker can take, and a play left out here is one the registry refused -- recorded on the
    battle with its reason, which the console reports every time.
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
        if card_id == MIRROR_CARD_ID or (decks is not None and card_id not in decks.get(side, ())):
            battle.untracked_plays.add((side, card_id))
            continue
        spec = specs.get(card_id)
        if spec is None or spec.elixir_cost is None:
            battle.untracked_plays.add((side, card_id))
            continue
        form_code = int(play.get('form_code') or 0)
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
                  'effective_cost': float(spec.elixir_cost), 'form_code': form_code,
                  'native_form_code': form_code, 'source': 'native_command_queue'}))
    events.sort(key=lambda event: (event.tick, event.owner or 0))
    return tuple(events)


def build(frame: dict, health: dict, episode_id: str, deduced: dict | None = None,
          revealed: dict | None = None, battle: Battle | None = None,
          reserved: float = 0.0, plays=None, decks: dict | None = None,
          hand_forms: dict | None = None, pending_ability_sources=(),
          evo_required: dict | None = None):
    """One ObservationV1 for the local actor, FAIR tier.

    Pass the same Battle for every frame of a battle: velocity, entity identity, age, the
    clock and the crown count all need continuity, and without it the policy sees a board
    where nothing moves, nothing has history and no tower has ever fallen.

    hand_forms: card id -> form for our hand (see action_mask).
    """
    from native_runner.contracts import (PLAYER_RUNTIME_SEMANTIC_FIELDS, SemanticEvidenceLevel,
                                         ENTITY_RUNTIME_SEMANTIC_FIELDS,
                                         TOWER_RUNTIME_SEMANTIC_FIELDS,
                                         CausalGroupKind, CausalGroupRefV1, EntityStateV1,
                                         ObservationTier, ObservationV1, PlayerStateV1,
                                         TimeStateV1, TowerStateV1)

    revealed = revealed or {}
    side = health.get('local_side')
    if battle is None or battle.episode_id != episode_id:
        battle = Battle(episode_id)
    tick = battle.tick(frame['game_tick'])
    live = {(e['x'], e['y']): e for e in frame['entities'] if is_tower(e)}

    phase, multiplier = timeline().phase(tick)
    remaining = max(0, gameplay_end_tick() - tick) * TICK_MS

    towers = []
    tower_rows: list = []
    id_by_address: dict[int, int] = {}
    crowns = {0: 0, 1: 0}
    destroyed: set[int] = set()
    tower_ids: dict[int, list[int]] = {0: [], 1: []}
    for index, (x, y, owner, kind) in enumerate(TOWERS):
        found = live.get((x, y))
        entity_id = 5000000 + index
        tower_ids[owner].append(entity_id)
        if found and float(found.get('hp', 0)) >= 0:
            battle.tower_seen.add(index)
            if float(found.get('max_hp') or 1) > 0:
                battle.tower_max[index] = float(found.get('max_hp') or 1)
            hitpoints = float(found.get('hp', 0))
            battle.tower_hp[index] = hitpoints
        elif found:
            # Health unreadable this frame (the reader gives -1): keep the last reading. A
            # glitch must not read as a destroyed tower -- that is a crown, and in overtime the
            # end of the battle.
            hitpoints = battle.tower_hp.get(index, battle.tower_max.get(index, 1.0))
        else:
            # Absent means destroyed only if we have seen it alive in this battle; before
            # that it is simply a tower we have not resolved, and must not score a crown.
            hitpoints = 0.0 if index in battle.tower_seen else float(battle.tower_max.get(index, 1))
        maximum = battle.tower_max.get(index, 1.0) or 1.0
        active = hitpoints > 0
        if not active and index in battle.tower_seen:
            crowns[1 - owner] += 1
            destroyed.add(index)
        tower_rows.append((found, dict(
            entity_id=entity_id, owner=owner, tower_kind=kind,
            position=(float(x), float(y)),
            tower_troop_id=None if kind == 'king' else TOWER_PRINCESS,
            hitpoints=hitpoints, max_hitpoints=maximum, active=active)))
        if found and _addr(found.get('address')) is not None:
            id_by_address[_addr(found['address'])] = entity_id

    battle.towers_down = destroyed
    for index in list(battle.tower_down_tick):
        if index not in destroyed:
            del battle.tower_down_tick[index]
    for index in destroyed:
        battle.tower_down_tick.setdefault(index, tick)

    archetypes = archetype_by_card()
    entities = []
    addresses: set[str] = set()
    def resolve(e: dict):
        """Archetype for a board object. Its own data id first (what the object IS -- the unit
        a spawner made, the projectile of a spell); the card mapping only as the fallback for
        recordings made before the reader carried data ids. A form FirstLight never had (an
        evolution released after 15.535) is shown as its base unit rather than dropped."""
        if e.get('data_id'):
            found = archetype_by_data(e['data_id'])
            if found is not None:
                return found
        if is_effect(e) or e.get('card_id', -1) == -1:
            return None
        card_id = e['card_id']
        found = archetypes.get(card_id)
        if found is None and base_card(card_id) != card_id:
            found = archetypes.get(base_card(card_id))
            if found is not None:
                battle.form_fallbacks.add(card_id)
        return found

    for e in frame['entities']:
        if not is_tower(e) and resolve(e) is not None and _addr(e.get('address')) is not None:
            known_id, _age, _birth = battle.identify(str(e['address']), tick)
            id_by_address[_addr(e['address'])] = known_id

    def target_of(e: dict) -> tuple[int | None, bool]:
        raw_target = _addr(e.get('target') or '0x0') or 0
        if raw_target == 0:
            return None, True
        found_id = id_by_address.get(raw_target)
        return found_id, found_id is not None

    for e in frame['entities']:
        if is_tower(e):
            continue
        if not is_effect(e) and (e.get('hp', 0) < 0 or e.get('max_hp', 0) < 0):
            # Health unreadable this frame (a building being placed or torn down has read -1
            # for one frame). Leaving the unit out for a frame costs little; passing -1 makes
            # the strict contract reject the whole observation, and the turn with it.
            battle.unreadable_hp += 1
            continue
        resolved = resolve(e)
        if resolved is None:
            # Spell area effects are the usual case: our reader reports the spell's card id,
            # which has no entity archetype, and a strict tensorizer raises on it. Leaving the
            # entity out loses a short-lived effect; sending it in loses every decision while
            # it is on the board.
            key = e['card_id'] if e['card_id'] != -1 else -int(e.get('data_id') or 0)
            battle.unresolved[key] = battle.unresolved.get(key, 0) + 1
            continue
        global_id, kind = resolved
        address = str(e.get('address') or f"{e['x']}:{e['y']}:{e['card_id']}")
        addresses.add(address)
        entity_id, age_ms, birth = battle.identify(address, tick)
        if _addr(address) is not None:
            id_by_address[_addr(address)] = entity_id
        if is_effect(e):
            # A shot or spell in flight: position, velocity and what it is. It has no
            # hitpoints, attack, movement or deploy state of its own.
            card = e['card_id'] if e['card_id'] != -1 else None
            velocity = battle.velocity(address, tick, e['x'], e['y'])
            projectile = (projectile_state(e, global_id, entity_id, velocity, tick)
                          if kind == 'projectile' else None)
            entities.append(EntityStateV1(
                native_data_global_id=global_id, entity_id=entity_id, owner=e['side'],
                card_id=card, entity_kind=kind,
                position=(float(e['x']), float(e['y'])),
                velocity=velocity, projectile_state=projectile,
                age_ms=age_ms, visible=True,
                causal_group=CausalGroupRefV1(
                    kind=(CausalGroupKind.VOLLEY if kind == 'projectile'
                          else CausalGroupKind.PERSISTENT_EFFECT),
                    handle=f"{e['side']}:{global_id}:{birth}", source_card_id=card),
                runtime_provenance=runtime_provenance(ENTITY_RUNTIME_SEMANTIC_FIELDS, None,
                                                      None, None, tick, projectile)))
            continue
        target_id, target_known = target_of(e)
        attack, movement, deployment = troop_runtime(e, entity_id, tick, target_id,
                                                     target_known, _hit_speed(card_of_unit(global_id, e['card_id'])))
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
    own_abilities: tuple = ()
    for p in frame['players']:
        deck = p.get('deck_card_ids') or []
        # Only the actor's own hand is private state it may see. After settlement this
        # client exposes both hands; the opponent's must still go in as public-only.
        # Our hand is ours to read even mid-draw, when one slot (possibly slot 0) is -1.
        readable = p['side'] == side and any(i >= 0 for i in p['hand_deck_indices'])
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
            runtime_by_slot = {str(pos): {'form_code': int((hand_forms or {}).get(deck[i], 0))}
                               for pos, i in enumerate(hand_slots)
                               if deck and 0 <= i < len(deck)}
            abilities, evolutions = own_runtime_states(p, entities, side, tick, battle,
                                                       evo_required)
            players.append(PlayerStateV1(
                elixir_exact=own_elixir,
                hand=hand, next_card=nxt, deck=tuple(deck), cycle=cycle,
                private_state_visible=True,
                metadata={'hand_slot_by_card': slot_by_card,
                          'hand_runtime_by_slot': runtime_by_slot},
                ability_runtime_states=abilities, evolution_runtime_states=evolutions,
                runtime_provenance=_state_provenance(
                    PLAYER_RUNTIME_SEMANTIC_FIELDS,
                    {**({'ability_runtime_states': (SemanticEvidenceLevel.NATIVE_DERIVED,
                                                    ('player+0x3a0/+0x3a8 controllers',))}
                        if abilities else {}),
                     **({'evolution_runtime_states': (SemanticEvidenceLevel.NATIVE_DERIVED,
                                                      ('player+0x2e8 progress vector',))}
                        if evolutions else {})}, tick, ()),
                **common))
            own_abilities = abilities
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
        action_mask=action_mask(frame, side, own_elixir, reserved, towers, hand_forms,
                                own_abilities, pending_ability_sources),
        episode_id=episode_id,
        ruleset_id=ruleset_id())
    return observation, battle
