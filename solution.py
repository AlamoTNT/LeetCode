"""
股票交易问题
动态规划解法总结
"""


# maxProfit_{}: {}中是交易次数

def maxProfit_1(prices):
    """
    只进行一次交易
    实时维护当前最小的买入价格/最大的利润
    :param prices:
    :return:
    """
    # prices = [7,1,5,3,6,4]
    min_price, max_profit = float('inf'), 0
    for price in prices:
        max_profit = max(max_profit, price - min_price)
        min_price = min(min_price, price)
    return max_profit

print(maxProfit_1([7,1,5,3,6,4]))


def maxProfit_1_dp(prices):
    """
    动态规划解法
    只进行一次交易, 状态: 当前天结束时进行过的交易次数和手上是否持有股票
    dp[i][0][0]/dp[i][0][1]/dp[i][1][0]
    简化成dp[i][0]/dp[i][1]/dp[i][2]
    :param prices:
    :return:
    """
    if not prices: return 0
    dp = [[0, 0, 0] for _ in range(len(prices))]
    dp[0][0], dp[0][1], dp[0][2] = 0, -prices[0], 0
    for i in range(1, len(prices)):
        dp[i][0] = 0
        dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])
        dp[i][2] = max(dp[i-1][2], dp[i-1][1]+prices[i])
    return dp[-1][2]

print(maxProfit_1_dp([7,1,5,3,6,4]))


def maxProfit_n(prices):
    """
    可进行多次交易, 不限交易次数
    
    :param prices:
    :return:
    """
