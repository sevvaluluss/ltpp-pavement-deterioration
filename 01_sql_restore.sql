-- ============================================================
-- LTPP Pavement Deterioration Under Climate Variability
-- Script 1: SQL Server Database Restore
-- Author: Metehan Alp Memis
-- Journal: Transportation Geotechnics (TRGEO-D-26-01076)
-- ============================================================

-- Step 1: Check logical file names in BAK files
RESTORE FILELISTONLY 
FROM DISK = N'C:\LTPP\Bucket_141968.bak';

RESTORE FILELISTONLY 
FROM DISK = N'C:\LTPP\Bucket_141969.bak';

RESTORE FILELISTONLY 
FROM DISK = N'C:\LTPP\Bucket_141971.bak';

-- Step 2: Restore main dataset (IRI, Traffic, Performance)
RESTORE DATABASE [LTPP_MAIN]
FROM DISK = N'C:\LTPP\Bucket_141968.bak'
WITH
    MOVE N'Bucket_141968' TO N'C:\LTPP\LTPP_MAIN.mdf',
    MOVE N'Bucket_141968_log' TO N'C:\LTPP\LTPP_MAIN_log.ldf',
    REPLACE, STATS = 10;

-- Step 3: Restore MERRA climate dataset
RESTORE DATABASE [LTPP_MERRA]
FROM DISK = N'C:\LTPP\Bucket_141969.bak'
WITH
    MOVE N'Bucket_141969' TO N'C:\LTPP\LTPP_MERRA.mdf',
    MOVE N'Bucket_141969_log' TO N'C:\LTPP\LTPP_MERRA_log.ldf',
    REPLACE, STATS = 10;

-- Step 4: Restore materials dataset
RESTORE DATABASE [LTPP_MATERIAL]
FROM DISK = N'C:\LTPP\Bucket_141971.bak'
WITH
    MOVE N'Bucket_141971' TO N'C:\LTPP\LTPP_MATERIAL.mdf',
    MOVE N'Bucket_141971_log' TO N'C:\LTPP\LTPP_MATERIAL_log.ldf',
    REPLACE, STATS = 10;

-- Step 5: Verify tables in LTPP_MAIN
USE LTPP_MAIN;
SELECT TABLE_SCHEMA, TABLE_NAME 
FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_TYPE = 'BASE TABLE'
ORDER BY TABLE_NAME;

-- Step 6: Verify tables in LTPP_MERRA
USE LTPP_MERRA;
SELECT TABLE_NAME 
FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_TYPE = 'BASE TABLE'
ORDER BY TABLE_NAME;
