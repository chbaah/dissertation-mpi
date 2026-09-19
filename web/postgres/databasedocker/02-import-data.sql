COPY public.combined_prep_table (
    "row.names",
    collectiondate,
    country,
    countrycode,
    region,
    ntlmeanintensity,
    ntlcoverageperc,
    lcumixedforestperc,
    lcuclosedshrublandsperc,
    lcuopenshrublandsperc,
    lcuwoodysavannasperc,
    lcusavannasperc,
    lcugrasslandsperc,
    lcucroplandsperc,
    lcuurbanbuiltupperc,
    lcucroplandnatvegperc,
    lcubarrenperc,
    builtschools,
    builtmedicalfacilities,
    builtroads,
    builtresidence,
    mpi
)
FROM '/docker-entrypoint-initdb.d/data/ntl_lcu_osm_data.west.africa.csv'
WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ','
);
