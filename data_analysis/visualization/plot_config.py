# Connections between kinematic points:
CONNECTIONS = [
    # color, [(point_a, point_b), ...]
    ("black", [
        ("GPS_F", "Steering"),
        ("Steering", "GPS_R"),
    ]),
    ("blue", [
        ("P1l_F", "P2l_F"),
        ("P2l_F", "P4l_F"),
        ("P3l_F", "P4l_F"),
        ("P3l_F", "P1l_F"),
        ("P1l_R", "P2l_R"),
        ("P2l_R", "P4l_R"),
        ("P3l_R", "P4l_R"),
        ("P3l_R", "P5l_R"),
        ("P5l_R", "P1l_R"),
    ]),
    ("green", [
        ("P1l_F", "P7l_F"),
        ("P2l_F", "P7l_F"),
        ("P3l_F", "P6l_F"),
        ("P4l_F", "P6l_F"),
        ("P1l_R", "P7l_R"),
        ("P2l_R", "P7l_R"),
        ("P3l_R", "P6l_R"),
        ("P4l_R", "P6l_R"),
    ]),
    ("orange", [
        ("P5l_F", "P8l_F"),
        ("P5l_R", "P8l_R"),

    ]),
    ("red", [
        ("P10l_F", "P6l_F"),
        ("P6l_F", "P9l_F"),
        ("P8l_F", "P9l_F"),
        ("P7l_F", "P9l_F"),
        ("P10l_R", "P6l_R"),
        ("P6l_R", "P9l_R"),
        ("P8l_R", "P9l_R"),
        ("P7l_R", "P9l_R"),
    ]),
]