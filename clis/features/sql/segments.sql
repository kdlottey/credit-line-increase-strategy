-- Default rate and size by segment. `{by}` is validated against an allow-list in Python.
SELECT
    {by}                                                     AS segment,
    COUNT(*)                                                 AS accounts,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)       AS pct_of_book,
    ROUND(100.0 * AVG(default_next_month), 2)                AS default_rate_pct,
    ROUND(100.0 * AVG(util_now), 1)                          AS avg_util_pct
FROM accounts
GROUP BY {by}
ORDER BY {by}
