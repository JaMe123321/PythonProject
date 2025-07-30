CREATE TABLE daily_summary (
  id INT AUTO_INCREMENT PRIMARY KEY,
  date DATE NOT NULL,
  total_gherkin INT,
  level_S INT,
  level_A INT,
  level_B INT,
  level_C INT
);

show databases ;
use result;
select * from daily_summary