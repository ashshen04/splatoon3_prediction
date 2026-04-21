"""Mapping of stat.ink Splatoon 3 weapon codes to weapon classes.

The stat.ink CSV uses internal Splatoon 3 weapon codes (e.g. `sshooter`, `52gal`,
`nautilus47`), not human-readable names. This dict maps those codes to the
11 weapon classes.

Any weapon code not in this dict maps to "Unknown".
"""

WEAPON_CLASSES: dict[str, str] = {
    # ============================================================= Shooters
    "wakaba": "Shooter",                    # Splattershot Jr.
    "momiji": "Shooter",                    # Custom Splattershot Jr.
    "sshooter": "Shooter",                  # Splattershot
    "sshooter_collabo": "Shooter",          # Tentatek Splattershot
    "heroshooter_replica": "Shooter",       # Hero Shot Replica
    "ordershooter_replica": "Shooter",      # Order Shot Replica
    "52gal": "Shooter",                     # .52 Gal
    "52gal_deco": "Shooter",                # .52 Gal Deco
    "96gal": "Shooter",                     # .96 Gal
    "96gal_deco": "Shooter",                # .96 Gal Deco
    "prime": "Shooter",                     # Splattershot Pro
    "prime_collabo": "Shooter",             # Forge Splattershot Pro
    "jetsweeper": "Shooter",                # Jet Squelcher
    "jetsweeper_custom": "Shooter",         # Custom Jet Squelcher
    "promodeler_mg": "Shooter",             # Aerospray MG
    "promodeler_rg": "Shooter",             # Aerospray RG
    "sharp": "Shooter",                     # Splash-o-matic
    "sharp_neo": "Shooter",                 # Neo Splash-o-matic
    "l3reelgun": "Shooter",                 # L-3 Nozzlenose
    "l3reelgun_d": "Shooter",               # L-3 Nozzlenose D
    "h3reelgun": "Shooter",                 # H-3 Nozzlenose
    "h3reelgun_d": "Shooter",               # H-3 Nozzlenose D
    "bold": "Shooter",                      # N-ZAP '85
    "bold_neo": "Shooter",                  # N-ZAP '89
    "spaceshooter": "Shooter",              # Splattershot Nova
    "spaceshooter_collabo": "Shooter",      # Annaki Splattershot Nova
    "squiclean_a": "Shooter",               # Squeezer
    "squiclean_b": "Shooter",               # Foil Squeezer

    # ============================================================= Blasters
    "hotblaster": "Blaster",                # Blaster
    "hotblaster_custom": "Blaster",         # Custom Blaster
    "longblaster": "Blaster",               # Range Blaster
    "longblaster_custom": "Blaster",        # Custom Range Blaster
    "nova": "Blaster",                      # Luna Blaster
    "nova_neo": "Blaster",                  # Luna Blaster Neo
    "rapid": "Blaster",                     # Rapid Blaster
    "rapid_deco": "Blaster",                # Rapid Blaster Deco
    "rapid_elite": "Blaster",               # Rapid Blaster Pro
    "rapid_elite_deco": "Blaster",          # Rapid Blaster Pro Deco
    "clashblaster": "Blaster",              # Clash Blaster
    "clashblaster_neo": "Blaster",          # Clash Blaster Neo
    "sblast91": "Blaster",                  # S-BLAST '91
    "sblast92": "Blaster",                  # S-BLAST '92

    # ============================================================= Rollers
    "splatroller": "Roller",                # Splat Roller
    "splatroller_collabo": "Roller",        # Krak-On Splat Roller
    "heroroller_replica": "Roller",         # Hero Roller Replica
    "orderroller_replica": "Roller",        # Order Roller Replica
    "dynamoroller": "Roller",               # Dynamo Roller
    "dynamoroller_tesla": "Roller",         # Gold Dynamo Roller
    "carbonroller": "Roller",               # Carbon Roller
    "carbonroller_deco": "Roller",          # Carbon Roller Deco
    "variableroller": "Roller",             # Flingza Roller
    "variableroller_foil": "Roller",        # Foil Flingza Roller
    "wideroller": "Roller",                 # Big Swig Roller
    "wideroller_collabo": "Roller",         # Big Swig Roller Express

    # ============================================================= Brushes
    "pablo": "Brush",                       # Inkbrush
    "pablo_hue": "Brush",                   # Inkbrush Nouveau
    "hokusai": "Brush",                     # Octobrush
    "hokusai_hue": "Brush",                 # Octobrush Nouveau
    "bigbrush": "Brush",                    # Painbrush
    "bigbrush_nouveau": "Brush",            # Painbrush Nouveau
    "orderbrush_replica": "Brush",          # Order Brush Replica

    # ============================================================= Chargers
    "splatcharger": "Charger",              # Splat Charger
    "splatcharger_collabo": "Charger",      # Z+F Splat Charger
    "herocharger_replica": "Charger",       # Hero Charger Replica
    "ordercharger_replica": "Charger",      # Order Charger Replica
    "splatscope": "Charger",                # Splatterscope
    "splatscope_collabo": "Charger",        # Z+F Splatterscope
    "liter4k": "Charger",                   # E-liter 4K
    "liter4k_custom": "Charger",            # Custom E-liter 4K
    "liter4k_scope": "Charger",             # E-liter 4K Scope
    "liter4k_scope_custom": "Charger",      # Custom E-liter 4K Scope
    "bamboo14mk1": "Charger",               # Bamboozler 14 Mk I
    "bamboo14mk2": "Charger",               # Bamboozler 14 Mk II
    "soytuber": "Charger",                  # Goo Tuber
    "soytuber_custom": "Charger",           # Custom Goo Tuber
    "snipewriter5h": "Charger",             # Snipewriter 5H
    "snipewriter5b": "Charger",             # Snipewriter 5B

    # ============================================================= Splatlings
    "splatspinner": "Splatling",            # Mini Splatling
    "splatspinner_collabo": "Splatling",    # Zink Mini Splatling
    "barrelspinner": "Splatling",           # Heavy Splatling
    "barrelspinner_deco": "Splatling",      # Heavy Splatling Deco
    "herospinner_replica": "Splatling",     # Hero Splatling Replica
    "orderspinner_replica": "Splatling",    # Order Splatling Replica
    "hydra": "Splatling",                   # Hydra Splatling
    "hydra_custom": "Splatling",            # Custom Hydra Splatling
    "kugelschreiber": "Splatling",          # Ballpoint Splatling
    "kugelschreiber_hue": "Splatling",      # Ballpoint Splatling Nouveau
    "nautilus47": "Splatling",              # Nautilus 47
    "nautilus79": "Splatling",              # Nautilus 79
    "barrelspinner_edit": "Splatling",      # Heavy Edit Splatling
    "barrelspinner_edit_custom": "Splatling",  # Heavy Edit Splatling Nouveau

    # ============================================================= Dualies
    "maneuver": "Dualie",                   # Splat Dualies
    "maneuver_collabo": "Dualie",           # Enperry Splat Dualies
    "heromaneuver_replica": "Dualie",       # Hero Dualie Replicas
    "ordermaneuver_replica": "Dualie",      # Order Dualie Replicas
    "sputtery": "Dualie",                   # Dapple Dualies
    "sputtery_hue": "Dualie",               # Dapple Dualies Nouveau
    "quadhopper_black": "Dualie",           # Dark Tetra Dualies
    "quadhopper_white": "Dualie",           # Light Tetra Dualies
    "kelvin525": "Dualie",                  # Glooga Dualies
    "kelvin525_deco": "Dualie",             # Glooga Dualies Deco
    "dualsweeper": "Dualie",                # Dualie Squelchers
    "dualsweeper_custom": "Dualie",         # Custom Dualie Squelchers
    "douserdualies_ff": "Dualie",           # Douser Dualies FF
    "douserdualies_nt": "Dualie",           # Douser Dualies NT

    # ============================================================= Brellas
    "parashelter": "Brella",                # Splat Brella
    "parashelter_sorella": "Brella",        # Sorella Brella
    "heroshelter_replica": "Brella",        # Hero Brella Replica
    "ordershelter_replica": "Brella",       # Order Brella Replica
    "campingshelter": "Brella",             # Tenta Brella
    "campingshelter_sorella": "Brella",     # Tenta Sorella Brella
    "spygadget": "Brella",                  # Undercover Brella
    "spygadget_sorella": "Brella",          # Undercover Sorella Brella
    "24shelter_a": "Brella",                # Recycled Brella 24 Mk I
    "24shelter_b": "Brella",                # Recycled Brella 24 Mk II

    # ============================================================= Sloshers
    "bucketslosher": "Slosher",             # Slosher
    "bucketslosher_deco": "Slosher",        # Slosher Deco
    "heroslosher_replica": "Slosher",       # Hero Slosher Replica
    "orderslosher_replica": "Slosher",      # Order Slosher Replica
    "hissen": "Slosher",                    # Tri-Slosher
    "hissen_hue": "Slosher",                # Tri-Slosher Nouveau
    "screwslosher": "Slosher",              # Sloshing Machine
    "screwslosher_neo": "Slosher",          # Sloshing Machine Neo
    "bottlegeyser": "Slosher",              # Bloblobber
    "bottlegeyser_foil": "Slosher",         # Bloblobber Deco
    "washtub": "Slosher",                   # Explosher
    "washtub_custom": "Slosher",            # Custom Explosher

    # ============================================================= Stringers
    "tristringer": "Stringer",              # Tri-Stringer
    "tristringer_collabo": "Stringer",      # Inkline Tri-Stringer
    "herostringer_replica": "Stringer",     # Hero Stringer Replica
    "orderstringer_replica": "Stringer",    # Order Stringer Replica
    "lact450": "Stringer",                  # REEF-LUX 450
    "lact450_deco": "Stringer",             # REEF-LUX 450 Deco
    "fullcharge": "Stringer",               # Wellstring V
    "fullcharge_custom": "Stringer",        # Custom Wellstring V

    # ============================================================= Splatanas
    "drainbrush": "Splatana",               # Splatana Wiper
    "drainbrush_deco": "Splatana",          # Splatana Wiper Deco
    "jimuwiper": "Splatana",                # Splatana Stamper (fallback code)
    "jimuwiper_hue": "Splatana",            # Neo Splatana Stamper (fallback)
    "stamper": "Splatana",                  # Splatana Stamper
    "stamper_hue": "Splatana",              # Neo Splatana Stamper
    "herosplatana_replica": "Splatana",     # Hero Splatana Replica
    "ordersplatana_replica": "Splatana",    # Order Splatana Replica
}

WEAPON_CLASS_LIST = [
    "Shooter", "Blaster", "Roller", "Brush", "Charger",
    "Splatling", "Dualie", "Brella", "Slosher", "Stringer", "Splatana",
]


def get_weapon_class(weapon_code: str) -> str:
    """Return the class for a weapon code, defaulting to 'Unknown'."""
    if weapon_code is None:
        return "Unknown"
    return WEAPON_CLASSES.get(weapon_code, "Unknown")
