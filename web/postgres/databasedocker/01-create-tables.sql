CREATE TABLE IF NOT EXISTS combined_prep_table (
    id BIGSERIAL PRIMARY KEY,
    country TEXT,
    countrycode TEXT,
    region TEXT,
    collectiondate DATE,

    ntlmeanintensity DOUBLE PRECISION,
    ntlcoverageperc DOUBLE PRECISION,

    lcumixedforestperc DOUBLE PRECISION,
    lcuclosedshrublandsperc DOUBLE PRECISION,
    lcuopenshrublandsperc DOUBLE PRECISION,
    lcuwoodysavannasperc DOUBLE PRECISION,
    lcusavannasperc DOUBLE PRECISION,
    lcugrasslandsperc DOUBLE PRECISION,
    lcucroplandsperc DOUBLE PRECISION,
    lcuurbanbuiltupperc DOUBLE PRECISION,
    lcucroplandnatvegperc DOUBLE PRECISION,
    lcubarrenperc DOUBLE PRECISION,

    builtschools DOUBLE PRECISION,
    builtmedicalfacilities DOUBLE PRECISION,
    builtroads DOUBLE PRECISION,
    builtresidence DOUBLE PRECISION,

    mpi DOUBLE PRECISION
);
