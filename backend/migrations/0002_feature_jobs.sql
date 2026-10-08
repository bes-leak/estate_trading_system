-- очередь задач на расчёт средовых признаков для новых домов.
-- отдельный брокер не нужен: воркеры разбирают строки через FOR UPDATE SKIP LOCKED

CREATE TABLE IF NOT EXISTS feature_jobs (
    id           BIGSERIAL PRIMARY KEY,
    building_id  BIGINT NOT NULL REFERENCES buildings (id) ON DELETE CASCADE,
    status       TEXT NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending', 'running', 'done', 'failed')),
    attempts     SMALLINT NOT NULL DEFAULT 0,
    last_error   TEXT,
    created_at   TIMESTAMP NOT NULL DEFAULT now(),
    updated_at   TIMESTAMP NOT NULL DEFAULT now(),

    -- на один дом не больше одной незакрытой задачи
    UNIQUE (building_id)
);

-- воркер выбирает задачи по статусу и возрасту, индекс только по нужным строкам
CREATE INDEX IF NOT EXISTS feature_jobs_pending_idx
    ON feature_jobs (created_at)
    WHERE status IN ('pending', 'running');

-- признак того, что у дома уже посчитаны средовые признаки:
-- пока NULL, оценка использует значения ближайшего дома
ALTER TABLE buildings ADD COLUMN IF NOT EXISTS features_computed_at TIMESTAMP;

-- у домов из первоначальной загрузки признаки уже есть
UPDATE buildings
SET features_computed_at = now()
WHERE features_computed_at IS NULL
  AND distance_metro IS NOT NULL;
