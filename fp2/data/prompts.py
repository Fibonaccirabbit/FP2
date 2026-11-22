"""Deterministic natural-language descriptions of preference vectors.

``preference_to_text`` turns a preference vector on the probability simplex
(2 or 3 non-negative weights summing to 1) into a short English sentence:

* Template category: the gap between the weights (|w1 - w2| for two
  objectives; the gaps between the sorted weights for three) selects one of
  ten templates, from "perfect equilibrium" to "extreme focus".
* Objective names: each environment has fixed objective names; the dominant
  objective fills the primary slot and the others the secondary slots.
* Wording: an adverb is chosen from a fixed list indexed by the preference
  geometry, and a few phrases are intensified or softened depending on how
  concentrated the weights are.

The mapping is a pure function of its input (no randomness, no hashing).
"""
import numpy as np

ENV_OBJECTIVES = {
    "MO-Ant-v2": ["x_forward_distance", "y_lateral_distance"],
    "MO-HalfCheetah-v2": ["forward_speed", "energy_efficiency"],
    "MO-Hopper-v2": ["forward_speed", "jump_height"],
    "MO-Hopper-v3": ["forward_speed", "jump_height", "energy_efficiency"],
    "MO-Swimmer-v2": ["forward_speed", "energy_efficiency"],
    "MO-Walker2d-v2": ["forward_speed", "energy_efficiency"],
}

_TWO_OBJ_TEMPLATES = {
    'perfectly_balanced': "{adv}, perfect equilibrium between {obj1} and {obj2} with equal importance",
    'nearly_balanced': "{adv}, near balance of {obj1} and {obj2} under equal regard for",
    'minimal_preference': "{adv}, minimal preference for {primary} over {secondary} with regard for",
    'slight_preference': "{adv}, slight favor for {primary} while maintaining {secondary}",
    'light_preference': "{adv}, light priority on {primary} alongside regard for {secondary}",
    'moderate_preference': "{adv}, moderate emphasis on {primary} relative to {secondary}",
    'notable_preference': "{adv}, notable tilt toward {primary} with clear regard for {secondary}",
    'strong_preference': "{adv}, strong bias toward {primary} against {secondary}",
    'very_strong_preference': "{adv}, very strong dedication to {primary} over {secondary}",
    'extreme_preference': "{adv}, extreme focus on {primary} neglecting {secondary}",
}

# 'nearly_balanced' never occurs for three objectives (see _three_objective_text).
_THREE_OBJ_TEMPLATES = {
    'perfectly_balanced': "{adv}, perfect equilibrium among {obj1}, {obj2}, and {obj3} with equal importance",
    'minimal_preference': "{adv}, minimal preference for {primary} over {secondary} and {tertiary} with regard for",
    'slight_preference': "{adv}, slight favor for {primary} while maintaining {secondary} and {tertiary}",
    'light_preference': "{adv}, light priority on {primary} alongside regard for {secondary} and {tertiary}",
    'moderate_preference': "{adv}, moderate emphasis on {primary} relative to {secondary} and {tertiary}",
    'notable_preference': "{adv}, notable tilt toward {primary} with clear regard for {secondary} and {tertiary}",
    'strong_preference': "{adv}, strong bias toward {primary} against {secondary} and {tertiary}",
    'very_strong_preference': "{adv}, very strong dedication to {primary} over {secondary} and {tertiary}",
    'extreme_preference': "{adv}, extreme focus on {primary} neglecting {secondary} and {tertiary}",
}

# Two objectives: the adverb index is int(50 * |w1 - w2| / sqrt(2)) + (0..2),
# which never exceeds 37 on the simplex, so only the first 38 entries are used.
_OBJ1_ADVERBS = [
    "decisively", "primarily", "predominantly", "chiefly", "mainly",
    "principally", "fundamentally", "essentially", "centrally", "critically",
    "vitally", "crucially", "pivotally", "strategically", "systematically",
    "methodically", "deliberately", "intentionally", "purposefully", "specifically",
    "particularly", "especially", "notably", "remarkably", "distinctively",
    "characteristically", "typically", "consistently", "persistently", "steadily",
    "firmly", "solidly", "robustly", "strongly", "powerfully",
    "intensely", "vigorously", "energetically",
]
_OBJ2_ADVERBS = [
    "alternatively", "secondarily", "supplementarily", "additionally", "complementarily",
    "supportively", "auxiliarily", "subsidiarily", "ancillarily", "peripherally",
    "marginally", "tangentially", "indirectly", "implicitly", "subtly",
    "delicately", "gently", "softly", "mildly", "lightly",
    "moderately", "reasonably", "sensibly", "prudently", "cautiously",
    "carefully", "thoughtfully", "considerately", "respectfully", "diplomatically",
    "tactfully", "skillfully", "artfully", "elegantly", "gracefully",
    "smoothly", "fluidly", "seamlessly",
]
_THREE_OBJ_ADVERBS = [
    "strategically", "systematically", "thoughtfully", "carefully", "deliberately",
    "purposefully", "methodically", "precisely", "effectively", "efficiently",
    "optimally", "intelligently", "wisely", "skillfully", "expertly",
]

# Centroid and vertices of the 3-objective simplex.
_THREE_OBJ_ANCHORS = [
    np.array([1 / 3, 1 / 3, 1 / 3]),
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
]


def _gap_category(gap):
    """Map a weight gap in [0, 1] to one of the ten template categories."""
    if gap < 0.05:
        return 'perfectly_balanced'
    elif gap < 0.08:
        return 'nearly_balanced'
    elif gap < 0.12:
        return 'minimal_preference'
    elif gap < 0.18:
        return 'slight_preference'
    elif gap < 0.25:
        return 'light_preference'
    elif gap < 0.35:
        return 'moderate_preference'
    elif gap < 0.45:
        return 'notable_preference'
    elif gap < 0.6:
        return 'strong_preference'
    elif gap < 0.8:
        return 'very_strong_preference'
    return 'extreme_preference'


def _two_objective_text(pref, objectives):
    w1, w2 = float(pref[0]), float(pref[1])
    angle = np.arctan2(w2, w1) * 180 / np.pi  # degrees, in [0, 90]
    concentration = max(w1, w2) / (w1 + w2)

    # Below 45 degrees objective 1 dominates; at or above, objective 2.
    if angle < 45:
        adverbs, primary, secondary = _OBJ1_ADVERBS, objectives[0], objectives[1]
    else:
        adverbs, primary, secondary = _OBJ2_ADVERBS, objectives[1], objectives[0]
    diagonal_distance = abs(w1 - w2) / np.sqrt(2)
    adverb = adverbs[int(diagonal_distance * 50) + int(angle / 10) % 3]

    text = _TWO_OBJ_TEMPLATES[_gap_category(abs(w1 - w2))].format(
        adv=adverb, obj1=objectives[0], obj2=objectives[1],
        primary=primary, secondary=secondary)

    if concentration > 0.85:
        text = text.replace("with", "with pronounced").replace("while", "while distinctly")
    elif concentration < 0.6:
        text = text.replace("over", "compared to").replace("against", "relative to")
    return text


def _three_objective_text(pref, objectives):
    order = np.argsort(pref)[::-1]  # indices by descending weight
    w = pref[order]
    gap_12, gap_23, gap_13 = w[0] - w[1], w[1] - w[2], w[0] - w[2]

    if gap_13 < 0.08:
        category = 'perfectly_balanced'
    elif gap_12 < 0.05 and gap_23 >= 0.08:  # two leaders, one clearly behind
        category = 'notable_preference' if gap_23 > 0.15 else 'slight_preference'
    elif gap_23 < 0.05 and gap_12 >= 0.08:  # one leader, two close followers
        category = 'strong_preference' if gap_12 > 0.15 else 'moderate_preference'
    else:
        category = _gap_category(gap_13)  # gap_13 >= 0.08 here

    adverb = _THREE_OBJ_ADVERBS[int(gap_13 * 15) % 15]
    text = _THREE_OBJ_TEMPLATES[category].format(
        adv=adverb, obj1=objectives[0], obj2=objectives[1], obj3=objectives[2],
        primary=objectives[order[0]], secondary=objectives[order[1]],
        tertiary=objectives[order[2]])

    variance = np.var(pref)
    if variance > 0.05:
        text = text.replace("with", "with pronounced").replace("while", "while distinctly")
    elif variance < 0.02:
        text = text.replace("over", "compared to").replace("against", "relative to")

    # Close to the centroid or a vertex: sharpen the qualifier.
    if min(0.4 * np.linalg.norm(pref - anchor) for anchor in _THREE_OBJ_ANCHORS) < 0.1:
        if "moderate" in text:
            text = text.replace("moderate", "distinctly moderate")
        elif "slight" in text:
            text = text.replace("slight", "clearly slight")
    return text


def preference_to_text(preference, env_name):
    """Describe a 2- or 3-objective preference vector for ``env_name`` in English."""
    pref = np.array(preference, dtype=float)
    objectives = ENV_OBJECTIVES[env_name]
    if len(pref) == 2:
        return _two_objective_text(pref, objectives)
    if len(pref) == 3:
        return _three_objective_text(pref, objectives)
    raise ValueError(f"expected 2 or 3 objectives, got {len(pref)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        raise SystemExit("usage: python -m fp2.prompts <env> <w1> <w2> [w3]   e.g. python -m fp2.prompts MO-Hopper-v2 0.7 0.3")
    print(preference_to_text([float(w) for w in sys.argv[2:]], sys.argv[1]))
