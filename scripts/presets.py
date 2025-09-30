ALEPH_PRESETS = {
    "mushroom": {
        "dt": {
            "sniper": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 40).\n"
                    ":- aleph_set(nodes, 80000).\n"
                    ":- aleph_set(evalfn, laplace).\n"
                    ":- aleph_set(clauselength, 6).\n"
                    ":- aleph_set(minacc, 0.90).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 1).\n"
                )
            },
            "sweet_spot": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 64).\n"
                    ":- aleph_set(nodes, 100000).\n"
                    ":- aleph_set(evalfn, wracc).\n"
                    ":- aleph_set(clauselength, 8).\n"
                    ":- aleph_set(minacc, 0.80).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 5).\n"
                )
            },
            "sweeper": {
                "settings": (
                    ":- aleph_set(search, bf).\n"
                    ":- aleph_set(openlist, 1000).\n"
                    ":- aleph_set(nodes, 150000).\n"
                    ":- aleph_set(evalfn, coverage).\n"
                    ":- aleph_set(clauselength, 4).\n"
                    ":- aleph_set(minacc, 0.70).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 20).\n"
                )
            },
        },
        "rf": {
            "sniper": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 40).\n"
                    ":- aleph_set(nodes, 80000).\n"
                    ":- aleph_set(evalfn, laplace).\n"
                    ":- aleph_set(clauselength, 6).\n"
                    ":- aleph_set(minacc, 0.90).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 1).\n"
                )
            },
            "sweet_spot": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 64).\n"
                    ":- aleph_set(nodes, 100000).\n"
                    ":- aleph_set(evalfn, wracc).\n"
                    ":- aleph_set(clauselength, 8).\n"
                    ":- aleph_set(minacc, 0.80).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 5).\n"
                )
            },
            "sweeper": {
                "settings": (
                    ":- aleph_set(search, bf).\n"
                    ":- aleph_set(openlist, 1000).\n"
                    ":- aleph_set(nodes, 150000).\n"
                    ":- aleph_set(evalfn, coverage).\n"
                    ":- aleph_set(clauselength, 4).\n"
                    ":- aleph_set(minacc, 0.70).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 20).\n"
                )
            },
        },
        "xgb": {
            "sniper": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 40).\n"
                    ":- aleph_set(nodes, 80000).\n"
                    ":- aleph_set(evalfn, laplace).\n"
                    ":- aleph_set(clauselength, 6).\n"
                    ":- aleph_set(minacc, 0.90).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 1).\n"
                )
            },
            "sweet_spot": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 64).\n"
                    ":- aleph_set(nodes, 100000).\n"
                    ":- aleph_set(evalfn, wracc).\n"
                    ":- aleph_set(clauselength, 8).\n"
                    ":- aleph_set(minacc, 0.80).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 5).\n"
                )
            },
            "sweeper": {
                "settings": (
                    ":- aleph_set(search, bf).\n"
                    ":- aleph_set(openlist, 1000).\n"
                    ":- aleph_set(nodes, 150000).\n"
                    ":- aleph_set(evalfn, coverage).\n"
                    ":- aleph_set(clauselength, 4).\n"
                    ":- aleph_set(minacc, 0.70).\n"
                    ":- aleph_set(minpos, 5).\n"
                    ":- aleph_set(noise, 20).\n"
                )
            },
        },
    },
    "adult": {
        "dt": {
            "sniper": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 60).\n"
                    ":- aleph_set(nodes, 100000).\n"
                    ":- aleph_set(evalfn, laplace).\n"
                    ":- aleph_set(clauselength, 4).\n"
                    ":- aleph_set(noise, 200).\n"
                    ":- aleph_set(minacc, 0.80).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
            "sweet_spot": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 80).\n"
                    ":- aleph_set(nodes, 120000).\n"
                    ":- aleph_set(evalfn, wracc).\n"
                    ":- aleph_set(clauselength, 3).\n"
                    ":- aleph_set(noise, 400).\n"
                    ":- aleph_set(minacc, 0.70).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
            "sweeper": {
                "settings": (
                    ":- aleph_set(search, bf).\n"
                    ":- aleph_set(openlist, 1500).\n"
                    ":- aleph_set(nodes, 200000).\n"
                    ":- aleph_set(evalfn, coverage).\n"
                    ":- aleph_set(clauselength, 2).\n"
                    ":- aleph_set(noise, 1400).\n"
                    ":- aleph_set(minacc, 0.60).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
        },
        "rf": {
            "sniper": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 60).\n"
                    ":- aleph_set(nodes, 100000).\n"
                    ":- aleph_set(evalfn, laplace).\n"
                    ":- aleph_set(clauselength, 6).\n"
                    ":- aleph_set(noise, 200).\n"
                    ":- aleph_set(minacc, 0.80).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
            "sweet_spot": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 80).\n"
                    ":- aleph_set(nodes, 120000).\n"
                    ":- aleph_set(evalfn, wracc).\n"
                    ":- aleph_set(clauselength, 4).\n"
                    ":- aleph_set(noise, 400).\n"
                    ":- aleph_set(minacc, 0.70).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
            "sweeper": {
                "settings": (
                    ":- aleph_set(search, bf).\n"
                    ":- aleph_set(openlist, 1500).\n"
                    ":- aleph_set(nodes, 200000).\n"
                    ":- aleph_set(evalfn, coverage).\n"
                    ":- aleph_set(clauselength, 3).\n"
                    ":- aleph_set(noise, 1400).\n"
                    ":- aleph_set(minacc, 0.60).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
        },
        "xgb": {
            "sniper": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 60).\n"
                    ":- aleph_set(nodes, 100000).\n"
                    ":- aleph_set(evalfn, laplace).\n"
                    ":- aleph_set(clauselength, 6).\n"
                    ":- aleph_set(noise, 200).\n"
                    ":- aleph_set(minacc, 0.80).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
            "sweet_spot": {
                "settings": (
                    ":- aleph_set(search, heuristic).\n"
                    ":- aleph_set(openlist, 80).\n"
                    ":- aleph_set(nodes, 120000).\n"
                    ":- aleph_set(evalfn, wracc).\n"
                    ":- aleph_set(clauselength, 4).\n"
                    ":- aleph_set(noise, 400).\n"
                    ":- aleph_set(minacc, 0.70).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
            "sweeper": {
                "settings": (
                    ":- aleph_set(search, bf).\n"
                    ":- aleph_set(openlist, 1500).\n"
                    ":- aleph_set(nodes, 200000).\n"
                    ":- aleph_set(evalfn, coverage).\n"
                    ":- aleph_set(clauselength, 3).\n"
                    ":- aleph_set(noise, 1400).\n"
                    ":- aleph_set(minacc, 0.60).\n"
                    ":- aleph_set(minpos, 5).\n"
                )
            },
        },
    },
}
