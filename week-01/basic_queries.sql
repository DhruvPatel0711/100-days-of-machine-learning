-- ============================================================
-- SQL Basics: Orientation Concepts
-- Source: notes.md — "Supervised vs Unsupervised Learning",
--         "Features and Labels"
-- ============================================================


-- ============================================================
-- CONCEPT 1: Supervised Learning Example
-- Definition: Supervised learning predicts a known label (y)
-- from features (X). Example from notes: Spam detection.
-- Formula: y = f(X)
-- ============================================================

CREATE TABLE emails (
    email_id INT PRIMARY KEY,
    word_count INT,
    has_link BOOLEAN,
    sender_reputation INT,   -- 0-100 score
    is_spam BOOLEAN          -- this is the LABEL (y)
);

INSERT INTO emails (email_id, word_count, has_link, sender_reputation, is_spam) VALUES
(1, 120, TRUE,  20, TRUE),
(2, 450, FALSE, 90, FALSE),
(3, 80,  TRUE,  10, TRUE),
(4, 600, FALSE, 95, FALSE),
(5, 95,  TRUE,  15, TRUE);

-- Query: Separate Features (X) from the Label (y)
SELECT
    word_count,          -- Feature
    has_link,            -- Feature
    sender_reputation,   -- Feature
    is_spam AS label_y   -- Label
FROM
    emails;


-- ============================================================
-- CONCEPT 2: Unsupervised Learning Example
-- Definition: Unsupervised learning finds hidden structure in
-- unlabeled data. Example from notes: Grouping news articles
-- without knowing their topics beforehand.
-- Formula: Structure = f(X)
-- ============================================================

CREATE TABLE articles (
    article_id INT PRIMARY KEY,
    word_count INT,
    avg_sentence_length DECIMAL,
    keyword_density DECIMAL
    -- Note: no label/target column exists — that's the point.
);

INSERT INTO articles (article_id, word_count, avg_sentence_length, keyword_density) VALUES
(1, 800,  18.2, 0.04),
(2, 1200, 22.5, 0.02),
(3, 750,  17.9, 0.05),
(4, 1500, 25.1, 0.01);

-- Query: There is no label to select — only features (X).
-- Unsupervised algorithms consume this directly to find structure.
SELECT
    word_count,
    avg_sentence_length,
    keyword_density
FROM
    articles;


-- ============================================================
-- CONCEPT 3: Features and Labels (Primary Example)
-- Definition: Features (X) are inputs, Labels (y) are outputs
-- we want to predict. Example from notes: House Price Prediction.
-- Features: square footage, bedrooms, zip code.
-- Label: price.
-- ============================================================

CREATE TABLE housing_data (
    id INT PRIMARY KEY,
    square_footage INT,
    num_bedrooms INT,
    zip_code VARCHAR(10),
    price DECIMAL
);

INSERT INTO housing_data (id, square_footage, num_bedrooms, zip_code, price) VALUES
(1, 1500, 3, '90210', 500000),
(2, 2000, 4, '90210', 750000),
(3, 800,  1, '10001', NULL),      -- Missing price (bad data)
(4, 3000, 5, '33101', 1200000);

-- Query 3a: Select clean Features (X) and Label (y)
SELECT
    square_footage,
    num_bedrooms,
    zip_code,
    price AS target_label
FROM
    housing_data
WHERE
    price IS NOT NULL   -- Crucial ML Step: Handle missing values
ORDER BY
    price DESC;

-- Query 3b: Isolate ONLY the Features (X) — what a model sees at inference time
SELECT
    square_footage,
    num_bedrooms,
    zip_code
FROM
    housing_data
WHERE
    price IS NOT NULL;

-- Query 3c: Isolate ONLY the Label (y) — what a model is trained to predict
SELECT
    price AS target_label
FROM
    housing_data
WHERE
    price IS NOT NULL;

-- Query 3d: Show the "bad data" row directly — this is WHY we filter
-- (Common Mistake from notes.md: forgetting to handle missing values)
SELECT
    *
FROM
    housing_data
WHERE
    price IS NULL;
