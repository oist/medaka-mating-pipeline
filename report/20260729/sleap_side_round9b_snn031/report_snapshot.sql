-- Exact SQL materialization of the bounded datasets embedded in artifact.json.
-- The upstream evidence remains the JSON files documented in REPORT_SOURCE_NOTES.md.
-- This file exists so every rendered report widget has an executable, auditable
-- source representation under the portable-report artifact contract.

CREATE TEMP VIEW production_overall AS
SELECT *
FROM (
    VALUES
        ('Round8', 0.9621848739495797, 0.6760373863273880, 0.6450523394540493, 229, 11, 7, 0.0375, 0.0, 5.229703412823346, 6.712812581797680, 'sleap-nn 0.3.1', 'legacy track max6'),
        ('Round9b epoch 32', 0.9663865546218487, 0.5952107438212129, 0.5704102961619956, 230, 10, 6, 0.041666666666666664, 0.0045871559633027525, 4.894811667111426, 6.000016808894983, 'sleap-nn 0.3.1', 'legacy track max6'),
        ('Fresh mOKS epoch 0', 0.9621848739495797, 0.5652972691163427, 0.5393878109485103, 229, 11, 7, 0.0375, 0.0, 5.114390732631103, 7.581858982017795, 'sleap-nn 0.3.1', 'legacy track max6')
) AS t(
    candidate, f1, matched_only_oks, per_gt_oks_missing_zero,
    matched, missed, extra, merge_rate, swap_rate,
    mouth_p95_px, tail_p95_px, runtime, path
);

CREATE TEMP VIEW production_per_experiment AS
SELECT *
FROM (
    VALUES
        ('K_1_1', 'Round8', 0.6844847500968708, 28, 'legacy track max6'),
        ('K_1_1', 'Round9b epoch 32', 0.6356529401814790, 28, 'legacy track max6'),
        ('K_1_1', 'Fresh mOKS epoch 0', 0.6219150672414441, 28, 'legacy track max6'),
        ('K_1_2', 'Round8', 0.6204814588932218, 26, 'legacy track max6'),
        ('K_1_2', 'Round9b epoch 32', 0.5518587428361363, 26, 'legacy track max6'),
        ('K_1_2', 'Fresh mOKS epoch 0', 0.4824059750254251, 26, 'legacy track max6'),
        ('K_1_3', 'Round8', 0.7209886694922680, 26, 'legacy track max6'),
        ('K_1_3', 'Round9b epoch 32', 0.5943071460574006, 26, 'legacy track max6'),
        ('K_1_3', 'Fresh mOKS epoch 0', 0.5848309067230594, 26, 'legacy track max6')
) AS t(experiment, candidate, matched_only_oks, image_count, path);

CREATE TEMP VIEW bootstrap_production AS
SELECT *
FROM (
    VALUES
        ('Round8', 0.6450523394540493, CAST(NULL AS DOUBLE PRECISION), CAST(NULL AS DOUBLE PRECISION), CAST(NULL AS DOUBLE PRECISION), 200000, 40),
        ('Round9b epoch 32', 0.5704102961619956, -0.07464204329205369, -0.10647432690285805, -0.04200544730604152, 200000, 40),
        ('Fresh mOKS epoch 0', 0.5393878109485103, -0.10566452850553909, -0.13711863342190175, -0.07414426741001837, 200000, 40)
) AS t(candidate, mean, delta_vs_round8, ci_low, ci_high, resamples, clusters);

CREATE TEMP VIEW parity AS
SELECT *
FROM (
    VALUES
        (3, 'Round8', 242, 0.0, 'yes', 'yes'),
        (3, 'Round9b epoch 32', 241, 0.0, 'yes', 'yes'),
        (6, 'Round8', 247, 0.0, 'yes', 'yes'),
        (6, 'Round9b epoch 32', 246, 0.0, 'yes', 'yes')
) AS t(max_instances, model, records, max_coordinate_delta_px, coordinates_equal, noncoordinate_exact);

CREATE TEMP VIEW training_checkpoints AS
SELECT *
FROM (
    VALUES
        (0, 551, 0.5937781333923340, 'yes', '3ac43c60c8e', 46, 0.0002466944570187479, 'sleap-nn 0.3.1'),
        (29, 16530, 0.5867544412612915, 'no', '80a94b9a394e', 46, 0.0002466944570187479, 'sleap-nn 0.3.1'),
        (78, 43529, 0.5847840309143066, 'no', 'feac8dbf666e', 46, 0.0002466944570187479, 'sleap-nn 0.3.1')
) AS t(epoch, global_step, moks, selected, sha256_short, best_val_loss_epoch, best_val_loss, runtime);

CREATE TEMP VIEW oracle_offset_stress AS
SELECT *
FROM (
    VALUES
        ('-8,-8', 'Round8', 0.2689984306764344, 10.840212556100415, 18.252090743778012, 74),
        ('-8,-8', 'Round9b epoch 32', 0.5554386138815581, 6.444092160887925, 9.175434688712789, 74),
        ('-8,-8', 'Fresh mOKS epoch 0', 0.5156988046672184, 8.363585394602545, 8.737387658258060, 74),
        ('0,0', 'Round8', 0.7514592670098673, 0.6495942541614613, 7.269395226007977, 74),
        ('0,0', 'Round9b epoch 32', 0.5908135730307763, 4.652051411076439, 5.835659666323884, 74),
        ('0,0', 'Fresh mOKS epoch 0', 0.5827570675584910, 4.483867682490735, 10.533959811547717, 74),
        ('8,8', 'Round8', 0.2691003672248694, 12.064315562975658, 7.274547116582075, 74),
        ('8,8', 'Round9b epoch 32', 0.6362865655259868, 4.715361488642023, 6.184560770016118, 74),
        ('8,8', 'Fresh mOKS epoch 0', 0.5560302483097094, 6.504070618062104, 12.851461278964027, 74)
) AS t(offset, candidate, matched_only_oks, mouth_p95_px, tail_p95_px, images);

-- Typical widget reads after the views above are materialized:
-- SELECT * FROM production_overall;
-- SELECT * FROM production_per_experiment;
-- SELECT * FROM bootstrap_production;
-- SELECT * FROM parity;
-- SELECT * FROM training_checkpoints;
-- SELECT * FROM oracle_offset_stress;
