from kennard_stone import train_test_split


def kennard_stone(
    X,
    y=None,
    test_size=None,
    train_size=None,
):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        train_size=train_size,
    )
