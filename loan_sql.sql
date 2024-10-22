CREATE DATABASE loan_dataset;

USE loan_dataset;

CREATE TABLE loan_data (
	customer_age INT,
	customer_income INT,
	home_ownership VARCHAR(10),
	employment_duration FLOAT,
	loan_intent VARCHAR(10),
	loan_grade VARCHAR(10),
	loan_amnt FLOAT,
	loan_int_rate FLOAT,
	term_years INT,
	historical_default VARCHAR(10),
	cred_hist_length INT,
	Current_loan_status  BOOLEAN
 );
 
-- 전체 고객 대출 불이행 비율 분석
SELECT 
    COUNT(*) AS total_loans,
    SUM(CASE WHEN Current_loan_status = FALSE THEN 1 ELSE 0 END) AS total_defaults,
    (SUM(CASE WHEN Current_loan_status = FALSE THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS default_rate
FROM loan_data;

-- 고객 연령대에 따른 불이행 비율 분석
SELECT 
    CASE 
        WHEN customer_age < 30 THEN '20대 이하'
        WHEN customer_age BETWEEN 30 AND 39 THEN '30대'
        WHEN customer_age BETWEEN 40 AND 49 THEN '40대'
        WHEN customer_age BETWEEN 50 AND 59 THEN '50대'
        ELSE '60대 이상'
    END AS age_group,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN Current_loan_status = 0 THEN 1 ELSE 0 END) AS total_defaults,
    (SUM(CASE WHEN Current_loan_status = 0 THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS default_rate
FROM loan_data
GROUP BY age_group;


-- 소득에 따른 불이행 비율 분석
SELECT 
    CASE 
        WHEN customer_income < 30000 THEN '저소득'
        WHEN customer_income BETWEEN 30000 AND 60000 THEN '중소득'
        ELSE '고소득'
    END AS income_group,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN Current_loan_status = 0 THEN 1 ELSE 0 END) AS total_defaults,
    (SUM(CASE WHEN Current_loan_status = 0 THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS default_rate
FROM loan_data
GROUP BY income_group;

-- 대출 요청 금액에 따른 불이행 비율 분석
SELECT 
    CASE 
        WHEN loan_amnt < 5000 THEN '소액 대출'
        WHEN loan_amnt BETWEEN 5000 AND 20000 THEN '중액 대출'
        ELSE '대액 대출'
    END AS loan_amount_group,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN Current_loan_status = 0 THEN 1 ELSE 0 END) AS total_defaults,
    (SUM(CASE WHEN Current_loan_status = 0 THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS default_rate
FROM loan_data
GROUP BY loan_amount_group;

-- 고용 기간에 따른 불이행 비율 분석
SELECT 
    CASE 
        WHEN employment_duration < 1 THEN '1년 미만'
        WHEN employment_duration BETWEEN 1 AND 3 THEN '1-3년'
        WHEN employment_duration BETWEEN 3 AND 5 THEN '3-5년'
        ELSE '5년 이상'
    END AS employment_duration_group,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN Current_loan_status = 0 THEN 1 ELSE 0 END) AS total_defaults,
    (SUM(CASE WHEN Current_loan_status = 0 THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS default_rate
FROM loan_data
GROUP BY employment_duration_group;


-- 신용 기록 길이에 따른 불이행 분석
SELECT 
    CASE 
        WHEN cred_hist_length < 1 THEN '1년 미만'
        WHEN cred_hist_length BETWEEN 1 AND 5 THEN '1-5년'
        ELSE '5년 이상'
    END AS credit_history_group,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN Current_loan_status = 0 THEN 1 ELSE 0 END) AS total_defaults,
    (SUM(CASE WHEN Current_loan_status = 0 THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS default_rate
FROM loan_data
GROUP BY credit_history_group;

-- 이자율에 따른 불이행 분석
SELECT 
    CASE 
        WHEN loan_int_rate < 10 THEN '10% 미만'
        WHEN loan_int_rate BETWEEN 10 AND 15 THEN '10%-15%'
        ELSE '15% 이상'
    END AS interest_rate_group,
    COUNT(*) AS total_loans,
    SUM(CASE WHEN Current_loan_status = 0 THEN 1 ELSE 0 END) AS total_defaults,
    (SUM(CASE WHEN Current_loan_status = 0 THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS default_rate
FROM loan_data
GROUP BY interest_rate_group;


 