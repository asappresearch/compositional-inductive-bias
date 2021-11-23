for checking:
select task_id, score, duration_seconds,completion_code, feedback from game_instance where status = 'COMPLETE' and completion_code like '%';

for analysis:
select task_id, score, duration_seconds, feedback from game_instance where status = 'COMPLETE' order by task_id;
select task_id, score, duration_seconds from game_instance where status = 'COMPLETE' order by task_id;
select task_id, num_holdout_correct, score, duration_seconds, feedback from game_instance where status ='COMPLETE' order by task_id;
select task_id, num_holdout_correct, score, duration_seconds from game_instance where status ='COMPLETE' order by task_id;

select replace(task_id, 'eng2', 'eng'), num_holdout_correct, score, duration_seconds from game_instance where status ='COMPLETE' and task_id like 'eng%' and start_datetime >= '20211119140000' and score > 0 order by replace(task_id, 'eng2', 'eng');

# eng
select task_id, avg(score), avg(num_holdout_correct), avg(duration_seconds) from (
    select replace(task_id, 'eng2', 'eng') as task_id, num_holdout_correct, score, duration_seconds from game_instance where status ='COMPLETE' and task_id like 'eng%' and start_datetime >= '20211119140000' and score > 0 order by replace(task_id, 'eng2', 'eng')
) group by task_id;

select task_id, count(1) as n, cast(avg(score)as int), avg(num_holdout_correct), cast(avg(duration_seconds)as int) from (
    select replace(replace(task_id, 'eng2', 'eng'), '_pnts', '') as task_id, num_holdout_correct, score, duration_seconds from game_instance
    where status ='COMPLETE' and
          task_id like 'eng%' and
          start_datetime >= '20211119140000' and
          score > 0
    order by replace(task_id, 'eng2', 'eng')
) group by task_id;

select task_id, count(1) as n, cast(avg(score)as int), avg(num_holdout_correct), cast(avg(duration_seconds)as int) from (
    select replace(replace(task_id, 'eng2', 'eng'), '_pnts', '') as task_id, num_holdout_correct, score, duration_seconds from game_instance
    where status ='COMPLETE' and
          task_id like 'eng_pnts_%' and
          start_datetime >= '20211119140000' and
          score > 0
    order by replace(task_id, 'eng2', 'eng')
) group by task_id;

# synth
select task_id, avg(score), avg(num_holdout_correct), avg(duration_seconds) from (
    select task_id, num_holdout_correct, score, duration_seconds from game_instance
    where status ='COMPLETE' and
          task_id not like 'eng%' and
          task_id like '%_autoinc_ho'and
          (start_datetime >= '20211119140000' or start_datetime <= '20211119010000')
    order by task_id
) group by task_id;

select task_id, count(1) as n, cast(avg(score) as int), avg(num_holdout_correct), cast(avg(duration_seconds) as int) from (
    select replace(replace(replace(task_id, '_autoinc_ho', ''), 'pnts_', ''), '3', '') as task_id,
    num_holdout_correct, score, duration_seconds from game_instance
    where status ='COMPLETE' and
          task_id not like 'eng%' and
          (task_id like '%_autoinc_ho' or task_id like 'pnts_%') and
          (start_datetime >= '20211119140000' or start_datetime <= '20211119010000') and
          score > 0
    order by task_id
) group by task_id;

select task_id, count(1) as n, cast(avg(score) as int), avg(num_holdout_correct), cast(avg(duration_seconds) as int) from (
    select replace(replace(replace(task_id, '_autoinc_ho', ''), 'pnts_', ''), '3', '') as task_id,
    num_holdout_correct, score, duration_seconds from game_instance
    where status ='COMPLETE' and
          task_id not like 'eng%' and
          (task_id like 'pnts_%') and
          (start_datetime >= '20211119140000' or start_datetime <= '20211119010000') and
          score > 0
    order by task_id
) group by task_id;

for feedbacks:
select task_id, score, duration_seconds, feedback from game_instance where status='COMPLETE';

# eng_proj was using rot, utnil 12:39am on 20 nov, ET :()
