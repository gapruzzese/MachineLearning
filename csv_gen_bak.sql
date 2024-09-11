\copy (
	WITH sampled_latest_proceedings AS (
	  SELECT *
	  FROM (
	    SELECT fp.*, 
	      CASE
		WHEN fp.hearing_date IS NOT NULL THEN ROW_NUMBER() OVER (PARTITION BY fp.idncase ORDER BY fp.hearing_date DESC)
		WHEN fp.input_date IS NOT NULL THEN ROW_NUMBER() OVER (PARTITION BY fp.idncase ORDER BY fp.input_date DESC)
		ELSE ROW_NUMBER() OVER (PARTITION BY fp.idncase ORDER BY fp.comp_date DESC)
	      END AS rn,
	      CASE
		WHEN fp.dec_code IN ('D', 'V', 'X') THEN false
		WHEN fp.dec_code IN ('A', 'E', 'G', 'R', 'U', 'Z', 'W') THEN true
		ELSE NULL
	      END AS is_stay
	    FROM foia_proceeding_04_24 fp
	  ) AS latest_proceedings
	  WHERE rn = 1 AND dec_code IS NOT NULL
	  ORDER BY RANDOM()
	  LIMIT (SELECT CAST(COUNT(*) * 0.01 AS INTEGER) FROM foia_proceeding_04_24)
	)
	SELECT 
	  fc.*,
	  slp.*,
	  CASE 
	    WHEN EXISTS (
	      SELECT 1 
	      FROM foia_rider_04_24 fr 
	      WHERE (fr.idnLeadCase = slp.idncase OR fr.idnRiderCase = slp.idncase) AND fr.blnActive = 1
	    ) THEN 1
	    ELSE 0
	  END AS is_family,
	  (
	    SELECT string_agg(fch.charge, '|')
	    FROM foia_charges_04_24 fch
	    WHERE fch.idncase = slp.idncase
	  ) AS charges,
	  (
	    SELECT string_agg(fa.appl_code, '|')
	    FROM foia_application_04_24 fa
	    WHERE fa.idncase = slp.idncase
	  ) AS applications
	FROM sampled_latest_proceedings slp
	JOIN foia_case_04_24 fc ON slp.idncase = fc.idncase
) TO '/mydata.csv' WITH (FORMAT CSV, HEADER);

