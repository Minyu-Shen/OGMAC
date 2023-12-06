import numpy as np

# given two poisson distribution with two means, numerically find the maximum order statistics of the two distributions


def find_order_stat_ratio(board_flow_rate, aligh_flow_rate):
    # rate in passengers per second
    maxis = []
    boars = []
    for i in range(20000):
        board_pax = np.random.poisson(board_flow_rate * 1.0)
        aligh_pax = np.random.poisson(aligh_flow_rate * 1.0)
        maxis.append(max(board_pax, aligh_pax))
        boars.append(board_pax)

    max_mean = sum(maxis) / len(maxis)
    board_mean = sum(boars) / len(boars)
    order_ratio = max_mean / board_mean
    return order_ratio


if __name__ == "__main__":
    order_ratio = find_order_stat_ratio(0.2, 0.1)
    print(order_ratio)
