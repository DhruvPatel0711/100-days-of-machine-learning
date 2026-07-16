-- ============================================================
-- Advanced SQL: Aggregations & Joins
-- Source: notes.md — "SQL Aggregations", "How It Works"
-- Concept: JOIN, GROUP BY, COUNT(), SUM(), AVG()
-- Goal: Combine fragmented relational data into a single, flat
-- structure, calculating statistical features for ML.
-- ============================================================


-- ============================================================
-- Setup: Shared Tables & Mock Data
-- ============================================================

CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    department_name VARCHAR(50),
    salary DECIMAL
);

CREATE TABLE performance_reviews (
    employee_id INT,
    review_score INT
);

INSERT INTO employees (employee_id, department_name, salary) VALUES
(101, 'Sales', 60000),
(102, 'Sales', 70000),
(103, 'Engineering', 120000),
(104, 'Engineering', 130000),
(105, 'HR', 55000);

INSERT INTO performance_reviews (employee_id, review_score) VALUES
(101, 8), (102, 9), (103, 10), (104, 7), (105, 9);


-- ============================================================
-- CONCEPT 1: COUNT() — row counts per group
-- Definition (notes.md): Grouping rows with the same value and
-- calculating summary statistics on them.
-- ============================================================

SELECT
    department_name,
    COUNT(employee_id) AS total_employees
FROM
    employees
GROUP BY
    department_name;


-- ============================================================
-- CONCEPT 2: SUM() — total salary cost per group
-- Example use case: total payroll burden per department.
-- ============================================================

SELECT
    department_name,
    SUM(salary) AS total_salary_cost
FROM
    employees
GROUP BY
    department_name;


-- ============================================================
-- CONCEPT 3: AVG() — average salary per group
-- Example from notes.md: "Finding the average salary per department."
-- ============================================================

SELECT
    department_name,
    AVG(salary) AS avg_salary
FROM
    employees
GROUP BY
    department_name;


-- ============================================================
-- CONCEPT 4: Combined Aggregation — COUNT + SUM + AVG together
-- Definition (notes.md visualization): imagine a spreadsheet
-- where all "Sales" rows are bundled together and their salary
-- column is averaged into a single new row.
-- ============================================================

SELECT
    department_name,
    COUNT(employee_id) AS total_employees,
    SUM(salary)         AS total_salary_cost,
    AVG(salary)         AS avg_salary
FROM
    employees
GROUP BY
    department_name
ORDER BY
    avg_salary DESC;


-- ============================================================
-- CONCEPT 5: JOIN — combining fragmented relational data
-- Goal (notes.md): Combine fragmented relational data into a
-- single, flat structure for predictive modeling.
-- ============================================================

SELECT
    e.employee_id,
    e.department_name,
    e.salary,
    p.review_score
FROM
    employees e
JOIN
    performance_reviews p ON e.employee_id = p.employee_id;


-- ============================================================
-- CONCEPT 6: JOIN + GROUP BY — aggregating across joined tables
-- Example use case: average performance review score per department,
-- built on the flat structure produced by the JOIN above.
-- ============================================================

SELECT
    e.department_name,
    COUNT(e.employee_id)  AS total_employees,
    AVG(e.salary)         AS avg_salary,
    AVG(p.review_score)   AS avg_review_score
FROM
    employees e
JOIN
    performance_reviews p ON e.employee_id = p.employee_id
GROUP BY
    e.department_name
ORDER BY
    avg_review_score DESC;
