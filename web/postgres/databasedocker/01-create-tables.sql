CREATE TABLE IF NOT EXISTS public.combined_prep_table
(
    "row.names" text COLLATE pg_catalog."default",
    collectiondate date,
    country text COLLATE pg_catalog."default",
    countrycode text COLLATE pg_catalog."default",
    region text COLLATE pg_catalog."default",
    ntlmeanintensity double precision,
    ntlcoverageperc double precision,
    lcumixedforestperc double precision,
    lcuclosedshrublandsperc double precision,
    lcuopenshrublandsperc double precision,
    lcuwoodysavannasperc double precision,
    lcusavannasperc double precision,
    lcugrasslandsperc double precision,
    lcucroplandsperc double precision,
    lcuurbanbuiltupperc double precision,
    lcucroplandnatvegperc double precision,
    lcubarrenperc double precision,
    builtschools integer,
    builtmedicalfacilities integer,
    builtroads integer,
    builtresidence integer,
    mpi double precision
);

ALTER TABLE IF EXISTS public.combined_prep_table
    OWNER TO postgres;
