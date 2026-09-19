// ============================================================================
// NODE.JS / EXPRESS BACKEND FOR THE MULTIDIMENSIONAL POVERTY INDEX APPLICATION
// This script provides the server side API used by the MPI web application.
// It connects to PostgreSQL, retrieves country, region and collection-year
// information, returns regional predictor data and provides a controlled proxy
// for retrieving GeoJSON data used by the choropleth map.
//
// Executable statements from the original script have not been changed in this
// commented version.
// ============================================================================

// Use ES module syntax to import the packages required by the backend server.
import express from 'express';
import { Pool } from 'pg';
import cors from 'cors';
import morgan from 'morgan';

const app = express();

// ---------------------------------------------------------
// Application configuration
// ---------------------------------------------------------

const PORT = Number(process.env.PORT || 3000);

const DB_HOST = process.env.DB_HOST || '127.0.0.1';
const DB_PORT = Number(process.env.DB_PORT || 5432);
const DB_NAME = process.env.DB_NAME || 'mpi';
const DB_USER = process.env.DB_USER || 'postgres';
const DB_PASSWORD = process.env.DB_PASSWORD || 'postgres';

console.log('Application configuration:');
console.log(`  PORT: ${PORT}`);
console.log(`  DB_HOST: ${DB_HOST}`);
console.log(`  DB_PORT: ${DB_PORT}`);
console.log(`  DB_NAME: ${DB_NAME}`);
console.log(`  DB_USER: ${DB_USER}`);



/*
 * ---------------------------------------------------------
 * Middleware
 * ---------------------------------------------------------
 */
// Configure the middleware used by the Express application.
// CORS allows requests from the frontend, express.json() enables JSON request
// handling and Morgan records HTTP requests in the server log.
app.use(cors());
app.use(express.json());
app.use(morgan('combined'))

// Create the PostgreSQL connection pool used by the API endpoints.
// The pool allows database connections to be reused across incoming requests.

/*
 * ---------------------------------------------------------
 * PostgreSQL connection pool
 * ---------------------------------------------------------
 */

const pool = new Pool({
    user: DB_USER,
    host: DB_HOST,
    database: DB_NAME,
    password: DB_PASSWORD,
    port: DB_PORT
});

// validation
//const requiredEnvVars = [
//    'DB_HOST',
//    'DB_NAME',
//    'DB_USER',
//    'DB_PASSWORD'
//];


// validation - Docker deployments to fail immediately when configuration is missing
// for (const variable of requiredEnvVars) {
//    if (!process.env[variable]) {
//        throw new Error(
//            `Required environment variable ${variable} is not configured`
//        );
//    }
//}

// Retrieve the list of West African countries stored in the supporting
// country table and return the result to the frontend as JSON.

app.get('/api/countries', async (req, res) => {
    try {
        //const result = await pool.query('SELECT country_name FROM west_africa_countries;');
        const result = await pool.query(`
		SELECT DISTINCT country AS country_name
            	FROM combined_prep_table
            	WHERE country IS NOT NULL
            	ORDER BY country_name;
		`);

        console.log(result.rows)
        res.json(result.rows); // ← this line is missing
    } catch (err) {
        console.error('Error executing query', err);
        res.status(500).json({ error: 'An error occurred while fetching data' });
    }
});


// Retrieve the distinct subnational regions available for the selected country.
// The country value is supplied as a query parameter by the frontend.

app.get('/api/regions', async (req, res) => {
    console.log('/api/regions route hit');
    console.log('Query params:', req.query);
    try {
        const selectedCountry = req.query.country; // Read the selected country from the URL query parameters.

        if (!selectedCountry) {
            return res.status(400).json({ error: 'Country parameter is required' });
        }

        // Use a parameterised query so the supplied values are passed separately from the SQL statement.
        const result = await pool.query(
            'select distinct(region) from combined_prep_table where country = $1',
            [selectedCountry]
        );

        console.log(result.rows);
        res.json(result.rows);

    } catch (err) {
        console.error('Error executing query', err);
        res.status(500).json({ error: 'An error occurred while fetching data' });
    }
});


// Retrieve the distinct collection years available for the selected country
// and regions. These years are used to populate the collection-year controls
// displayed on the web page.
app.get('/api/uniquedates', async (req, res) => {
    console.log('/api/uniquedates route hit');
    try {
        const selectedCountry = req.query.country;
        const regions = req.query.regions
            ? req.query.regions.split(',').map(r => r.trim())
            : [];

        if (!selectedCountry || regions.length === 0) {
            return res.status(400).json({ error: 'Country and regions are required' });
        }

        const result = await pool.query(
            `SELECT DISTINCT EXTRACT(YEAR FROM collectiondate) AS year
             FROM combined_prep_table
             WHERE country = $1
             AND region = ANY($2)
             ORDER BY year DESC`,
            [selectedCountry, regions]
        );

        console.log('dates result:', result.rows);
        res.json(result.rows);
        console.log('dates result:', result.rows);

    } catch (err) {
        console.error('Error fetching dates:', err);
        res.status(500).json({ error: 'An error occurred while fetching dates' });
    }
});


// Retain the alternative collection-year endpoint used during development.
// It performs a similar query but accepts the region values in a different form.
app.get('/api/uniquedatesoff', async (req, res) => {
    console.log('/api/uniquedates route hit');
    console.log('Query params:', req.query);
    try {
        const selectedCountry = req.query.country; // Read the selected country from the URL query parameters.
        let regions = req.query.regions

        if (!selectedCountry) {
            return res.status(400).json({ error: 'Country parameter is required' });
        }

        // Convert a single selected region to an array so the same query structure can be used for one or more regions.
        if (!Array.isArray(regions)) {
            regions = [regions];
        }

        // Use a parameterised query so the supplied values are passed separately from the SQL statement.
        const yearresult = await pool.query(
            `SELECT DISTINCT EXTRACT(YEAR FROM collectiondate) AS year
            FROM combined_prep_table
            WHERE country = $1
            AND region =  ANY($2)
            ORDER BY year
            `,
            [req.query.country, regions]
        );


        //console.log(yearresult.rows); // [{ year: 2019 }, { year: 2020 }, ...]

        console.log(yearresult.rows);
        res.json(yearresult.rows);

    } catch (err) {
        console.error('Error executing query', err);
        res.status(500).json({ error: 'An error occurred while fetching data' });
    }
});


// Retrieve the distinct regions associated with the supplied country code
// from the MPI table. This endpoint is retained as part of the backend API.
/*
app.get('/api/data', async (req, res) => {
  try {
    //console.log(req)
    //console.log("req.query:", req.query);
    const countrykey = Object.keys(req.query);
    const countryval = Object.values(req.query)[0];
    console.log("countryval", countryval);

    const client = await pool.connect();
    const query = "SELECT DISTINCT region FROM mpi WHERE countrycode = '" + countryval + "' ORDER BY region";
    const result = await client.query(query);
    console.log("result:", result.rows)
    client.release();
    res.json(result.rows);
  } catch (err) {
    console.error('Error executing query', err);
    res.status(500).json({ error: 'An error occurred while fetching data' });
  }
});
*/

app.get('/api/data', async (req, res) => {
    try {
        const countryval = Object.values(req.query)[0];

        const result = await pool.query(
            `
            SELECT DISTINCT region
            FROM mpi
            WHERE countrycode = $1
            ORDER BY region
            `,
            [countryval]
        );

        res.json(result.rows);

    } catch (err) {
        console.error('Error executing query', err);

        res.status(500).json({
            error: 'An error occurred while fetching data'
        });
    }
});


// Retrieve the most recent database record for each selected region.
// This endpoint returns the predictor data that can subsequently be supplied
// to the model prediction API.
app.get('/api/predict', async (req, res) => {
    // Read the selected regions supplied by the frontend.
    let regions = req.query.regions;

    // Convert a single selected region to an array so the same query structure can be used for one or more regions.
    if (!Array.isArray(regions)) {
        regions = [regions];
    }

    console.log('regions:', regions); // ['Ashanti', 'Northern', 'Greater Accra']

    // ANY() can be used with the region array when querying multiple selected regions.
    //const result = await pool.query(
    //    'SELECT * FROM combined_prep_table WHERE country = $1 AND region = ANY($2)',
    //    [req.query.country, regions]
    //);

    const result = await pool.query(
        `
        SELECT DISTINCT ON (region) *
        FROM combined_prep_table
        WHERE country = $1
        AND region = ANY($2)
        ORDER BY region, collectiondate DESC
        `,
        [req.query.country, regions]
    );

    res.json(result.rows);
});

/*
//api to convert shape file to geoJson

app.get('/api/plotgeojsondata', async (req, res) => {
    try {
        const features = [];
        const source = await shapefile.open('/home/charles/Documents/study/MscDataScience/Dissertation/Rstudio/dissertationdatasf/DHS/GHA/2019/GHA.shp');

        // loop through each feature in the shapefile
        while (true) {
            const result = await source.read();
            if (result.done) break;
            features.push(result.value);
        }

        // wrap features into a proper GeoJSON FeatureCollection
        const geojson = {
            type: 'FeatureCollection',
            features: features
        };

        res.json(geojson);

    } catch (err) {
        console.error('Error reading shapefile', err);
        res.status(500).json({ error: 'An error occurred while reading the shapefile' });
    }
});
*/
//const https = require('https');

// Provide a server-side proxy for retrieving GeoJSON from approved external
// sources. The proxy allows the frontend to request boundary data through this
// backend rather than directly from the external source.
app.get('/api/geojson', async (req, res) => {
    const url = req.query.url;

    if (!url) {
        return res.status(400).json({ error: 'url parameter is required' });
    }

    // Restrict GeoJSON requests to the external domains required by the application.
    const allowedDomains = [
        'geoboundaries.org',
        'geodata.ucdavis.edu',
        'raw.githubusercontent.com',
        'cdn.jsdelivr.net',
        'github.com'
    ];

    const isAllowed = allowedDomains.some(domain => url.includes(domain));
    if (!isAllowed) {
        return res.status(403).json({ error: 'Domain not allowed' });
    }

    try {
        const response = await fetch(url);

        if (!response.ok) {
            return res.status(response.status).json({
                error: `Upstream error: ${response.status}`
            });
        }

        const contentType = response.headers.get('content-type') || '';

        // Reject HTML responses because the endpoint expects GeoJSON or other JSON data.
        if (contentType.includes('text/html')) {
            return res.status(400).json({
                error: 'Upstream returned HTML not JSON'
            });
        }

        const text = await response.text();

        // Check the response structure before attempting to parse it as JSON.
        const trimmed = text.trim();
        if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) {
            return res.status(400).json({
                error: 'Upstream response is not JSON',
                preview: trimmed.substring(0, 100)
            });
        }

        // Parse the validated response and return it to the frontend as JSON.
        const geojson = JSON.parse(text);
        res.json(geojson);

    } catch (error) {
        console.error('GeoJSON proxy error:', error);
        res.status(500).json({ error: error.message });
    }
});


// Retrieve the distinct country names and their corresponding country codes.
// The frontend uses these values when constructing country-selection controls.
app.get('/api/countryshortcode', async (req, res) => {
    try {
        const result = await pool.query('SELECT DISTINCT country, countrycode FROM combined_prep_table ORDER BY country;');
        console.log(result.rows)
        res.json(result.rows); 
    } catch (err) {
        console.error('Error executing query', err);
        res.status(500).json({ error: 'An error occurred while fetching data' });
    }
});

// health check endpoint
app.get('/health', async (req, res) => {
    try {
        await pool.query('SELECT 1');

        res.status(200).json({
            status: 'healthy',
            database: 'connected'
        });

    } catch (error) {
        console.error('Health check failed:', error);

        res.status(503).json({
            status: 'unhealthy',
            database: 'disconnected'
        });
    }
});

// The MPI prediction itself is handled by the separate R Plumber API.
//app.get('')

//const PORT = process.env.POST || 5000;
// Start the Express server and listen for requests on the configured port.
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
