def min_cost_flights(N, s, f, K, flight_costs):
    dp = [[float('infinity') for _ in range(K + 1)] for _ in range(N)]
    dp[s][0] = 0

    for k in range(1, K + 1):
        for i in range(N):
            for j in range(N):
                if flight_costs[j][i] != -1 and dp[j][k-1] != float('infinity'):
                    dp[i][k] = min(dp[i][k], dp[j][k-1] + flight_costs[j][i])

    min_cost = min(dp[f][:K+1])

    return min_cost if min_cost != float('infinity') else -1

def main():
    N, s, f, K = map(int, input().split())
    flight_costs = []

    for _ in range(N):
        flight_costs.append(list(map(int, input().split())))

    result = min_cost_flights(N, s, f, K, flight_costs)
    print(result)

if __name__ == "__main__":
    main()
