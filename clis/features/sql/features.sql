-- Account-level features for one monthly snapshot.
-- Input view `snapshot` has month-aligned columns: s1..s5 (repayment status),
-- b1..b5 (statement balance), p1..p5 (amount paid); 1 = most recent month of the snapshot.
WITH base AS (
    SELECT
        *,
        b1 + b2 + b3 + b4 + b5                                   AS bill_sum,
        p1 + p2 + p3 + p4 + p5                                   AS paid_sum,
        b1 / NULLIF(limit_amt, 0)                                AS util_raw
    FROM snapshot
)
SELECT
    account_id, limit_amt, sex, age, education, marriage, default_next_month,
    s1 AS pay_status_1, s2 AS pay_status_2, b1 AS bill_1, p1 AS paid_1,
    GREATEST(util_raw, 0)                                        AS util_now,
    GREATEST(bill_sum / 5.0 / NULLIF(limit_amt, 0), 0)           AS util_avg,
    GREATEST(s1, s2, s3, s4, s5)                                 AS worst_delay,
    CAST(s1 > 0 AS INT) + CAST(s2 > 0 AS INT) + CAST(s3 > 0 AS INT)
        + CAST(s4 > 0 AS INT) + CAST(s5 > 0 AS INT)              AS months_delayed,
    -- share of balances repaid; accounts that owed nothing count as fully repaid
    LEAST(COALESCE(paid_sum / NULLIF(GREATEST(bill_sum, 0), 0), 1), 1)
                                                                 AS pay_ratio,
    (ABS(b1) + ABS(b2) + ABS(b3) + ABS(b4) + ABS(b5)) = 0        AS is_dormant,
    CASE WHEN (ABS(b1) + ABS(b2) + ABS(b3) + ABS(b4) + ABS(b5)) = 0
                             THEN '0. Dormant (no balance 5 mo)'
         WHEN s1 = -2        THEN '1. No spend this month'
         WHEN s1 = -1        THEN '2. Transactor (paid in full)'
         WHEN s1 = 0         THEN '3. Revolver'
         WHEN s1 = 1         THEN '4. 1 month late'
         ELSE                     '5. 2+ months late' END        AS customer_type,
    CASE WHEN b1 <= 0                THEN '0. No balance'
         WHEN util_raw < 0.10        THEN '1. <10%'
         WHEN util_raw < 0.30        THEN '2. 10-30%'
         WHEN util_raw < 0.50        THEN '3. 30-50%'
         WHEN util_raw < 0.80        THEN '4. 50-80%'
         WHEN util_raw <= 1.0        THEN '5. 80-100%'
         ELSE                             '6. Over limit' END    AS util_band
FROM base
