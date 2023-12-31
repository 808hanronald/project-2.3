-- MySQL Script generated by MySQL Workbench
-- Mon Oct  9 10:10:06 2023
-- Model: New Model    Version: 1.0
-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema project2
-- -----------------------------------------------------
DROP SCHEMA IF EXISTS `project2` ;

-- -----------------------------------------------------
-- Schema project2
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `project2` DEFAULT CHARACTER SET utf8 ;
USE `project2` ;

-- -----------------------------------------------------
-- Table `project2`.`ratings`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `project2`.`ratings` ;

CREATE TABLE IF NOT EXISTS `project2`.`ratings` (
  `tconst` VARCHAR(45) NOT NULL,
  `avg_rating` FLOAT NULL,
  `number_of_votes` VARCHAR(45) NULL,
  PRIMARY KEY (`tconst`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `project2`.`title_basics`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `project2`.`title_basics` ;

CREATE TABLE IF NOT EXISTS `project2`.`title_basics` (
  `tconst` VARCHAR(45) NOT NULL,
  `primary_title` VARCHAR(255) NULL,
  `start_year` CHAR(4) NULL,
  `run_time` VARCHAR(45) NULL,
  `ratings_tconst` VARCHAR(45) NULL,
  PRIMARY KEY (`tconst`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `project2`.`genres`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `project2`.`genres` ;

CREATE TABLE IF NOT EXISTS `project2`.`genres` (
  `genre_id` INT NOT NULL AUTO_INCREMENT,
  `genre_name` VARCHAR(45) NULL,
  PRIMARY KEY (`genre_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `project2`.`title_genres`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `project2`.`title_genres` ;

CREATE TABLE IF NOT EXISTS `project2`.`title_genres` (
  `title_basics_tconst` VARCHAR(45) NULL,
  `genres_genre_id` INT NULL)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
