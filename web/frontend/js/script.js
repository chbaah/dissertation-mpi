// ============================================================================
// MULTIDIMENSIONAL POVERTY INDEX WEB APPLICATION - FRONTEND JAVASCRIPT
// ============================================================================
// This script provides the main client-side functions used by the bulk MPI
// prediction page. It retrieves country, region and collection-year data from
// the Node.js backend, requests MPI predictions from the R Plumber API and
// presents the returned results in tables and interactive visualisations.
//
// The script also supports the bar chart, regional trend lines, poverty
// classification chart and choropleth map used to examine predicted MPI
// values. GeoJSON boundary data are retrieved and matched to the selected
// subnational regions for the spatial visualisation.
// ============================================================================


const NODE_API_URL = window.APP_CONFIG?.NODE_API_URL;
const PLUMBER_API_URL = window.APP_CONFIG?.PLUMBER_API_URL;

if (!NODE_API_URL) {
    throw new Error("NODE_API_URL is not configured");
}

if (!PLUMBER_API_URL) {
    throw new Error("PLUMBER_API_URL is not configured");
}

console.log("Node API:", NODE_API_URL);
console.log("Plumber API:", PLUMBER_API_URL);

// Store values that need to remain available across different page functions.
let selectedcountry = ''; 
let countryShortcodes = [];

// Obtain the current year for use in the page footer.
const CopyRightYearDate = new Date().getFullYear();

// Display the current year in the copyright section of the page.
copyRight = document.querySelector('.copyrightyear')
copyRight.innerText = '© ' + CopyRightYearDate + ' Copyright' ;

// Retrieve the available countries and their country codes from the Node.js backend.
async function loadCountryShortcodes() {

    try {

        const response = await fetch(`${NODE_API_URL}/api/countryshortcode`);

        if (!response.ok) {
            throw new Error('Failed to fetch country shortcodes');
        }

        const data = await response.json();

        // Store the returned country information so it can be reused by other page functions.
        countryShortcodes = data;

        console.log('Country shortcode data loaded');
        console.log(countryShortcodes);

    } catch (err) {

        console.error('Error loading country shortcodes:', err);

    }
}

// Return the country code associated with the selected country name.
function getCountryCode(countryName) {

    const result = countryShortcodes.find(
        item => item.country.toLowerCase() === countryName.toLowerCase()
    );

    return result ? result.countrycode : null;
}

// Convert the CSV response returned by the R Plumber API into JavaScript
// objects that can be used by the table and visualisation functions.
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.replace(/"/g, '').trim());

    return lines.slice(1).map(line => {
        const values = line.split(',').map(v => v.replace(/"/g, '').trim());
        const row = {};
        headers.forEach((header, i) => {
            row[header] = values[i];
        });
        return row;
    });
}


// Request predicted MPI values from the R Plumber API for the selected
// country, regions and collection year. = '2022-01-01'
async function fetchMPIPrediction(country, regions, year ) {

    try {
        // Convert the selected regions into the comma-separated format expected by the API.
        // ['Ashanti', 'Northern'] → 'ashanti,northern'
        const regionsString = regions.map(r => r.toLowerCase()).join(',');

        // Construct the Plumber API request using the selected country, regions and year.
        const url = `${PLUMBER_API_URL}/api/mpi?country=${encodeURIComponent(country)}&region=${encodeURIComponent(regionsString)}&year=${encodeURIComponent(year)}`;

        console.log('Calling R Plumber API:', url);

        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        // The Plumber endpoint returns CSV output, therefore read the response as
        // text before converting it into JavaScript objects.
        const csvText = await response.text();
        console.log('Raw CSV response:', csvText);

        // Convert the returned CSV text into an array of prediction records.
        const data = parseCSV(csvText);
        console.table(data);

        return data;

    } catch (error) {
        console.error('Error calling MPI API:', error);
    }
}


// Create the regional selection checkboxes returned for the selected country.
function renderRegionCheckboxes(regions) {
    
    const list = document.getElementById('checkboxList');
    const selregionlist = document.querySelector('.region-title');
    const submbutton = document.getElementById('submitBtn')

    // Update the regional-selection description displayed to the user.

    if (selregionlist){
        selregionlist.style.display = 'block';
    }

    // Update the regional-selection description displayed to the user.

    if (submbutton){
        submbutton.style.display = 'block';
    }


    list.innerHTML = '';
    regions.forEach((item, i) => {
        const label = document.createElement('label');
        label.className = 'checkbox-item';
        label.innerHTML = `
            <input type="checkbox" value="${item.region}" id="r${i}" />
            ${item.region}
        `;
        list.appendChild(label);
    });

    // Update the available collection years whenever the regional selection changes.
    list.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
        checkbox.addEventListener('change', loadDatesForCheckedRegions);
    });

}



// Retrieve the collection years available for the currently selected regions.
async function loadDatesForCheckedRegions() {
    const checked = [...document.querySelectorAll('#checkboxList input:checked')]
        .map(c => c.value);

    const dateSection = document.getElementById('dateSection');
    const dateList = document.getElementById('dateList');

    if (!dateSection) { console.error('#dateSection not found in HTML'); return; }
    if (!dateList)    { console.error('#dateList not found in HTML'); return; }

    // Hide the collection-year controls when no region has been selected.
    if (checked.length === 0) {
        dateSection.style.display = 'none';
        dateList.innerHTML = '';
        return;
    }

    try {
        const regionsParam = encodeURIComponent(checked.join(','));
        const response = await fetch(
            `${NODE_API_URL}/api/uniquedates?country=${encodeURIComponent(selectedcountry)}&regions=${regionsParam}`
        );

        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        const dates = await response.json();
        console.log('available dates:', dates);

        if (dates.length === 0) {
            dateList.innerHTML = '<span class="empty-msg">No dates available.</span>';
            dateSection.style.display = 'block';
            return;
        }

        // Extract the unique collection years returned for the selected regions.
        //const years = [...new Set(
        //    dates.map(d => new Date(d.year).getFullYear())
        //)].sort();

        const years = [...new Set(
            dates.map(d => parseInt(d.year))
            )].sort();

        console.log('unique years:', years);

        // Display the available collection years as selectable checkboxes.
        dateList.innerHTML = '';
        years.forEach((year, i) => {
            const label = document.createElement('label');
            label.className = 'checkbox-item';
            label.innerHTML = `
                <input type="checkbox" value="${year}" id="year${i}" />
                ${year}
            `;
            dateList.appendChild(label);
        });

        dateSection.style.display = 'block';

    } catch (error) {
        console.error('Error loading dates:', error);
    }
}



// Create the regional selection checkboxes returned for the selected country.


function renderTable(data) {
    if (!data || data.length === 0) return;

    const cols = Object.keys(data[0]);

    const headerHTML = cols.map(c => `<th>${c}</th>`).join('');

    const rowsHTML = data.map(row =>
        '<tr>' + cols.map(c => {
            const val = row[c];
            const numVal = Number(val);

            const cell = !isNaN(numVal) && !Number.isInteger(numVal)
                ? numVal.toFixed(3)
                : val ?? '';

            let className = '';

            if (c === 'mpi_predicted' && !isNaN(numVal)) {
                className = numVal < 0.3333 ? 'low-mpi' : 'high-mpi';
            }

            return `<td class="${className}">${cell}</td>`;
        }).join('') + '</tr>'
    ).join('');

    document.getElementById('tableContainer').innerHTML = `
        <div class="table-wrapper">
            <table class="styled-table">
                <thead><tr>${headerHTML}</tr></thead>
                <tbody>${rowsHTML}</tbody>
            </table>
        </div>
    `;

    document.getElementById('table-description').style.display = 'block';
    document.getElementById('tableSection').style.display = 'block';
    document.getElementById('tableContainer').style.display = 'block';
}



// Prepare the poverty-classification chart and allow the user to examine the
// Poor and Non-Poor distribution for each selected collection year.

function renderDonutChart(data) {

    // Identify the unique years available in the returned prediction data.
    const years = [...new Set(data.map(r => parseInt(r.year)))].sort();

    // Display a year selector when predictions are available for more than one year.
    const yearSelector = document.getElementById('donutYearSelect');
    if (yearSelector) {
        yearSelector.innerHTML = years.map(y =>
            `<option value="${y}" ${y === Math.max(...years) ? 'selected' : ''}>${y}</option>`
        ).join('');
        yearSelector.style.display = years.length > 1 ? 'inline-block' : 'none';

        // Update the classification chart when the user selects another year.
        yearSelector.onchange = () => renderDonutForYear(data, parseInt(yearSelector.value));
    }

    // Display the most recent available year when the chart is first shown.
    renderDonutForYear(data, Math.max(...years));
    document.getElementById('donutTab').style.display = 'block';
}

// Classify the predicted MPI values for one year using the study threshold and
// update the donut chart and summary metric cards.
function renderDonutForYear(data, year) {
    const yearData = data.filter(r => parseInt(r.year) === year);

    const poor    = yearData.filter(r => parseFloat(r.mpi_predicted) >= 0.3333).length;
    const nonPoor = yearData.filter(r => parseFloat(r.mpi_predicted) <  0.3333).length;
    const total   = poor + nonPoor;

    const poorPct    = total > 0 ? ((poor    / total) * 100).toFixed(1) : '0.0';
    const nonPoorPct = total > 0 ? ((nonPoor / total) * 100).toFixed(1) : '0.0';

    document.getElementById('donutTotal').textContent   = total;
    document.getElementById('donutNonPoor').textContent = `${nonPoor} — ${nonPoorPct}%`;
    document.getElementById('donutPoor').textContent    = `${poor} — ${poorPct}%`;

    const titleEl = document.getElementById('donutTitle');
    if (titleEl) titleEl.textContent = `Regional Distribution of Predicted MPI Poverty Classification (${year})`;

    if (window.donutChartInstance) window.donutChartInstance.destroy();

    const ctx = document.getElementById('donutCanvas').getContext('2d');
    window.donutChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Non-Poor (MPI < 0.333)', 'Poor (MPI ≥ 0.333)'],
            datasets: [{
                data:            [nonPoor, poor],
                backgroundColor: ['#EF9F27', '#E24B4A'],
                borderColor:     ['#BA7517', '#A32D2D'],
                borderWidth:      2,
                hoverOffset:      8
            }]
        },
        options: {
            responsive:          true,
            maintainAspectRatio: false,
            cutout:             '68%',
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            const t   = ctx.dataset.data.reduce((a, b) => a + b, 0);
                            const pct = ((ctx.parsed / t) * 100).toFixed(1);
                            return ` ${ctx.parsed} regions (${pct}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Display a grouped bar chart for comparing predicted MPI values across the
// selected regions and collection years.

function renderBarChart(data) {

    // Group the prediction records by collection year before constructing the Plotly traces.
    // result: { '2019': [...rows], '2020': [...rows], '2021': [...rows] }
    const groupedByYear = data.reduce((acc, row) => {
        const year = row.year || new Date(row.collectiondate).getFullYear();
        if (!acc[year]) acc[year] = [];
        acc[year].push(row);
        return acc;
    }, {});

    console.log('grouped by year:', groupedByYear);

    // Create one Plotly trace for each available collection year.
    const traces = Object.entries(groupedByYear).map(([year, rows]) => ({
        name: year,                          // shows in legend
        x: rows.map(r => r.region),          // x axis = regions
        y: rows.map(r => r.mpi_predicted),   // y axis = predicted MPI
        type: 'bar'
    }));

    const layout = {
        title: `Predicted MPI by Region - ${selectedcountry}`,
        barmode: 'group',                    // 'group' or 'stack'
        xaxis: { title: 'Region' },
        yaxis: { title: 'MPI Score', range: [0, 1] },
        shapes: [
            {
                type: 'line',
                x0: 0, x1: 1, xref: 'paper',
                y0: 0.3333, y1: 0.3333,
                line: { color: 'red', width: 1, dash: 'dash' }
            }
        ],
        annotations: [
            {
                x: 1, xref: 'paper',
                y: 0.3333,
                text: 'Poverty threshold (0.3333)',
                showarrow: false,
                font: { size: 11, color: 'red' },
                xanchor: 'right',
                yanchor: 'bottom'
            }
        ],
    };

    Plotly.newPlot('barChartContainer', traces, layout);

    // Display the chart section after the visualisation has been prepared.
    document.getElementById('chartSection').style.display = 'block';

    // Move the page to the chart section so the generated visualisation is visible.
    document.getElementById('chartSection')
    .scrollIntoView({ behavior: 'smooth' });
}

// Display regional trend lines showing changes in predicted MPI across the
// selected collection years.
function renderLineChart(data) {

    // Group the returned prediction records by region for the trend analysis.
    // result: { 'greater accra': [...rows], 'ashanti': [...rows] }
    const groupedByRegion = data.reduce((acc, row) => {
        const region = row.region;
        if (!acc[region]) acc[region] = [];
        acc[region].push(row);
        return acc;
    }, {});

    console.log('grouped by region:', groupedByRegion);

    // Create one Plotly trace for each region, using collection year on the
    // horizontal axis and predicted MPI on the vertical axis.
    const traces = Object.entries(groupedByRegion).map(([region, rows]) => {

        // Sort each region by year before drawing its MPI trend line.
        const sorted = rows.sort((a, b) => a.year - b.year);

        return {
            name: region,                            // shows in legend
            x: sorted.map(r => String(r.year)),      // x axis = years
            y: sorted.map(r => parseFloat(r.mpi_predicted)), // y axis = MPI
            type: 'scatter',
            mode: 'lines+markers',                   // line with dots at each year
            marker: { size: 8 },
            line: { width: 2 }
        };
    });

    console.log('traces:', traces);

    const layout = {
        title: `MPI Trend by Region — ${selectedcountry}`,
        xaxis: {
            title: 'Year',
            type: 'category'    // treat years as categories not continuous numbers
        },
        yaxis: {
            title: 'Predicted MPI Score',
            range: [0, 1]
        },
        // Add the study poverty threshold of 0.3333 as a reference line.
        shapes: [
            {
                type: 'line',
                x0: 0, x1: 1, xref: 'paper',
                y0: 0.3333, y1: 0.3333,
                line: { color: 'red', width: 1, dash: 'dash' }
            }
        ],
        annotations: [
            {
                x: 1, xref: 'paper',
                y: 0.3333,
                text: 'Poverty threshold (0.3333)',
                showarrow: false,
                font: { size: 11, color: 'red' },
                xanchor: 'right',
                yanchor: 'bottom'
            }
        ],
        legend: {
            orientation: 'h',       // horizontal legend below chart
            y: -0.2
        }
    };

    const config = {
        responsive: true            // chart resizes with the window
    };

    Plotly.newPlot('lineChartContainer', traces, layout, config);

        // Display the chart section after the visualisation has been prepared.
    document.getElementById('lineChartContainer').style.display = 'block';

    // Move the page to the chart section so the generated visualisation is visible.
    document.getElementById('lineChartContainer')
    .scrollIntoView({ behavior: 'smooth' });
}


// Render the scatter-chart function retained in the application for examining
// the relationship between the values supplied to this visualisation.
function renderScatterChart(data) {
    const traces = [{
        x: data.map(r => parseFloat(r.mpi)),
        y: data.map(r => parseFloat(r.mpi_predicted)),
        mode: 'markers',
        type: 'scatter',
        text: data.map(r => `${r.region} (${r.year})`),  // tooltip text
        hovertemplate: '%{text}<br>Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>',
        marker: { size: 8, color: '#378ADD', opacity: 0.7 }
    }];

    // perfect prediction line
    const allValues = [...data.map(r => parseFloat(r.mpi)), ...data.map(r => parseFloat(r.mpi_predicted))];
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);

    const perfectLine = [{
        x: [minVal, maxVal],
        y: [minVal, maxVal],
        mode: 'lines',
        type: 'scatter',
        name: 'Perfect prediction',
        line: { color: 'red', width: 1, dash: 'dash' }
    }];

    const layout = {
        title: `Predicted vs Actual MPI — ${selectedcountry}`,
        xaxis: { title: 'Actual MPI' },
        yaxis: { title: 'Predicted MPI' }
    };

    Plotly.newPlot('chartContainer', [...traces, ...perfectLine], layout, { responsive: true });
}



// ============================================================================
// CHOROPLETH MAP AND GEOJSON BOUNDARY MATCHING
// ============================================================================
// The following functions retrieve candidate administrative-boundary files,
// compare their regional names with the selected MPI regions and use the best
// available match for the choropleth map.

// Standardise regional names before comparing database and GeoJSON values.
function normaliseRegionName(name) {
    return name.toLowerCase()
               .trim()
               .replace(/\s+/g, ' ')       // collapse multiple spaces
               .replace(/[-_]/g, ' ');      // treat hyphens and underscores as spaces
}

function extractRegionName(feature) {
    const p = feature.properties;

    // Check common property fields used by different GeoJSON providers for region names.
    return p.shapeName   ||   // geoBoundaries
           p.NAME_1      ||   // GADM
           p.name        ||   // generic
           p.NAME        ||   // generic uppercase
           p.admin1Name  ||   // some UN sources
           p.ADM1_EN     ||   // some African sources
           p.REGION      ||   // some sources
           '';
}

// Measure how well the regional names in a candidate GeoJSON file correspond
// with the regions selected in the MPI application.
function scoreGeoJSON(geojson, selectedRegions) {
    const geoNames = geojson.features.map(f =>
        normaliseRegionName(extractRegionName(f))
    );

    console.log('GeoJSON region names found:', geoNames);

    const matched   = [];
    const unmatched = [];

    selectedRegions.forEach(region => {
        const normRegion = normaliseRegionName(region);

        const exactMatch = geoNames.includes(normRegion);
        const partialMatch = !exactMatch && geoNames.some(geoName =>
            geoName.includes(normRegion) || normRegion.includes(geoName)
        );

        if (exactMatch || partialMatch) {
            matched.push(region);
        } else {
            unmatched.push(region);
        }
    });

    return {
        matched,
        unmatched,
        matchCount:  matched.length,
        totalCount:  selectedRegions.length,
        isFullMatch: unmatched.length === 0,
        score:       matched.length / selectedRegions.length
    };
}
// Convert a GitHub file URL to its raw-content form when required.
function fixGithubRawUrl(url) {
    return url
        .replace('https://github.com/', 'https://raw.githubusercontent.com/')
        .replace('/raw/', '/');
}

// Build a list of candidate GeoJSON boundary sources for the selected country.
async function fetchAvailableVersions(countrycode) {
    const versions = [];

    // Add the geoBoundaries API as one candidate source of ADM1 boundaries.
    try {
        const apiUrl = `https://www.geoboundaries.org/api/current/gbOpen/${countrycode}/ADM1/`;
        const response = await fetch(apiUrl);

        if (response.ok) {
            const meta = await response.json();
            console.log('geoBoundaries API meta:', meta);

            if (meta.simplifiedGeometryGeoJSON) {
                versions.push({
                    label: `geoBoundaries current simplified (${meta.boundaryYear || 'latest'})`,
                    url:   meta.simplifiedGeometryGeoJSON,
                    year:  meta.boundaryYear || 9999
                });
            }

            if (meta.gjDownloadURL) {
                versions.push({
                    label: `geoBoundaries current full (${meta.boundaryYear || 'latest'})`,
                    url:   meta.gjDownloadURL,
                    year:  meta.boundaryYear || 9999
                });
            }
        }
    } catch (error) {
        console.warn('geoBoundaries API failed:', error);
    }

    // Add GADM as another candidate source of administrative boundaries.
    const gadmVersions = [
        {
            label: 'GADM 4.1 ADM1',
            url:   `https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_${countrycode}_1.json`,
            year:  2022
        },
        {
            label: 'GADM 3.6 ADM1',
            url:   `https://geodata.ucdavis.edu/gadm/gadm3.6/json/gadm36_${countrycode}_1.json`,
            year:  2018
        }
    ];
    versions.push(...gadmVersions);

    // Add the available Natural Earth boundary source hosted through the CDN.
    versions.push({
        label: 'Natural Earth (jsdelivr CDN)',
        url:   `https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json`,
        year:  2020
    });

    // Add the geoBoundaries CDN location as an alternative boundary source.
    const gbVersions = [
        {
            label: 'geoBoundaries via jsDelivr (latest)',
            url:   `https://cdn.jsdelivr.net/npm/geoboundaries@1.0.0/data/${countrycode}/ADM1/geoBoundaries-${countrycode}-ADM1_simplified.geojson`,
            year:  2023
        }
    ];
    versions.push(...gbVersions);

    // Add the available OCHA HDX boundary source as another candidate.
    const ochaMap = {
        GHA: 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_GHA_1.json',
        NGA: 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_NGA_1.json',
        SEN: 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_SEN_1.json',
        BEN: 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BEN_1.json',
        BFA: 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_BFA_1.json',
        MLI: 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_MLI_1.json',
        NER: 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_NER_1.json',
        TGO: 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_TGO_1.json',
        CIV: 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_CIV_1.json',
        GIN: 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_GIN_1.json',
        SLE: 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_SLE_1.json',
        LBR: 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_LBR_1.json',
        GMB: 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_GMB_1.json',
        MRT: 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_MRT_1.json',
        CPV: 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_CPV_1.json'
    };

    if (ochaMap[countrycode]) {
        versions.push({
            label: `GADM direct (${countrycode})`,
            url:    ochaMap[countrycode],
            year:   2022
        });
    }

    console.log(`Total versions to try for ${countrycode}:`, versions.length);
    return versions;
}

// Add historical boundary versions that may provide a better match for regional names.
async function fetchHistoricalVersions(countrycode) {
    const versions = [];

    // Include available GADM versions through the backend GeoJSON proxy.
    const gadmUrls = [
        {
            label: 'GADM 4.1 (2022 boundaries)',
            url:   `https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_${countrycode}_1.json`,
            year:  2022
        },
        {
            label: 'GADM 3.6 (2018 boundaries)',
            url:   `https://geodata.ucdavis.edu/gadm/gadm3.6/json/gadm36_${countrycode}_1.json`,
            year:  2018
        },
        {
            label: 'GADM 2.8 (2012 boundaries)',
            url:   `https://geodata.ucdavis.edu/gadm/gadm2.8/json/gadm28_${countrycode}_1.json`,
            year:  2012
        }
    ];

    versions.push(...gadmUrls);
    return versions;
}

// Retrieve one candidate GeoJSON file through the Node.js proxy endpoint.
async function downloadGeoJSON(url) {
    try {
        // Route the external request through the local backend proxy.
        const proxyUrl = `${NODE_API_URL}/api/geojson?url=${encodeURIComponent(url)}`;
        console.log(`  Downloading via proxy: ${url}`);

        const response = await fetch(proxyUrl);

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            console.warn(`  Proxy returned ${response.status}:`, err.error || '');
            return null;
        }

        const geojson = await response.json();

        // Confirm that the returned object contains a usable GeoJSON FeatureCollection.
        if (!geojson.features || !Array.isArray(geojson.features) || geojson.features.length === 0) {
            console.warn(`  Not a valid GeoJSON FeatureCollection`);
            return null;
        }

        console.log(`  Valid GeoJSON with ${geojson.features.length} features`);
        return geojson;

    } catch (error) {
        console.warn(`  Could not download GeoJSON:`, error.message);
        return null;
    }
}

// Compare the available boundary sources with the selected regions and return
// the GeoJSON file that provides the strongest regional-name match.
async function findBestGeoJSON(countrycode, selectedRegions) {

    console.log(`\n🔍 Searching for GeoJSON matching ${selectedRegions.length} regions...`);
    console.log('Selected regions:', selectedRegions);

    // update the loading message
    const updateStatus = (msg) => {
        const el = document.getElementById('mapStatus');
        if (el) el.textContent = msg;
    };

    // Combine the current and historical boundary sources that will be evaluated.
    updateStatus('Fetching available boundary versions...');
    const currentVersions    = await fetchAvailableVersions(countrycode);
    const historicalVersions = await fetchHistoricalVersions(countrycode);
    const allVersions        = [...currentVersions, ...historicalVersions];

    console.log(`Found ${allVersions.length} versions to try`);

    let bestResult = null;    // track best match found so far
    let bestScore  = -1;

    for (let i = 0; i < allVersions.length; i++) {
        const version = allVersions[i];

        updateStatus(`Checking version ${i + 1}/${allVersions.length}: ${version.label}...`);
        console.log(`\n Trying: ${version.label} — ${version.url}`);

        const geojson = await downloadGeoJSON(version.url);
        if (!geojson) {
            console.log(`   Skipping — download failed`);
            continue;
        }

        const result = scoreGeoJSON(geojson, selectedRegions);
        console.log(`  Matched: ${result.matchCount}/${result.totalCount}`);
        console.log(`  Matched regions:`, result.matched);
        console.log(`  Unmatched regions:`, result.unmatched);

        // Return immediately when all selected regions are matched.
        if (result.isFullMatch) {
            console.log(`  Perfect match found in: ${version.label}`);
            updateStatus(` Perfect match found: ${version.label}`);
            return {
                geojson,
                version,
                matched:   result.matched,
                unmatched: result.unmatched,
                score:     result.score,
                isPerfect: true
            };
        }

        // Retain the strongest partial match while the remaining sources are evaluated.
        if (result.score > bestScore) {
            bestScore  = result.score;
            bestResult = { geojson, version, ...result, isPerfect: false };
            console.log(`  💾 New best score: ${(result.score * 100).toFixed(0)}%`);
        }
    }

    // Use the strongest partial match when no source provides a complete match.
    if (bestResult) {
        console.log(`\n No perfect match found. Best was: ${bestResult.version.label}`);
        console.log(`   Score: ${(bestResult.score * 100).toFixed(0)}%`);
        console.log(`   Unmatched: ${bestResult.unmatched}`);
        updateStatus(` Best match: ${(bestResult.score * 100).toFixed(0)}% of regions found`);
        return bestResult;
    }

    // Return no boundary data when none of the candidate sources can be used.
    updateStatus(' Could not find any matching GeoJSON');
    return null;
}

// Create a regional MPI lookup using the most recent prediction available for each region.
function buildMpiLookupLatestYear(data) {

    // Track both the prediction and collection year while identifying the latest value.
    const grouped = {};

    data.forEach(row => {
        const key  = normaliseRegionName(row.region);
        const year = parseInt(row.year) || 0;
        const mpi  = parseFloat(row.mpi_predicted);

        if (!grouped[key]) {
            // first time seeing this region
            grouped[key] = { mpi, year };
        } else if (year > grouped[key].year) {
            // newer year found — update
            grouped[key] = { mpi, year };
        }
    });

    // Convert the grouped result into a simple region-to-MPI lookup.
    const lookup = {};
    Object.entries(grouped).forEach(([key, val]) => {
        lookup[key] = val.mpi;
    });

    console.log('MPI lookup (latest year per region):', lookup);
    return lookup;
}

// Render the choropleth map by matching predicted MPI values with the selected
// country's administrative boundaries.
async function renderChoroplethMap(data, countrycode) {

    const mapSection   = document.getElementById('mapTab');
    const mapWarning   = document.getElementById('mapWarning');
    const mapStatus    = document.getElementById('mapStatus');
    const mapContainer = document.getElementById('mapContainer');

    mapSection.style.display = 'block';
    mapWarning.style.display = 'none';
    mapContainer.innerHTML   = '';
    mapStatus.style.display  = 'block';
    mapStatus.textContent    = 'Starting map search...';

    const selectedRegions = [...new Set(data.map(r => r.region))];
    console.log('Regions to match:', selectedRegions);

    const result = await findBestGeoJSON(countrycode, selectedRegions);

    if (!result) {
        mapStatus.textContent = ' Could not load any map boundaries for this country.';
        return;
    }

    const { geojson, version, matched, unmatched, isPerfect } = result;

    mapStatus.style.display = 'none';

    if (!isPerfect && unmatched.length > 0) {
        mapWarning.style.display = 'block';
        mapWarning.innerHTML = `
            <strong> Partial match using: ${version.label}</strong><br>
            ${matched.length} of ${selectedRegions.length} regions matched.<br>
            Unmatched regions will appear grey:<br>
            <em>${unmatched.join(', ')}</em>
        `;
    }

    // Build the MPI lookup required for matching predictions to the map boundaries.
    const mpiLookup = buildMpiLookupLatestYear(data);

    // Store the collection year associated with each regional prediction for the hover text.
    const yearLookup = {};
    data.forEach(row => {
        const key  = normaliseRegionName(row.region);
        const year = parseInt(row.year) || 0;
        if (!yearLookup[key] || year > yearLookup[key]) {
            yearLookup[key] = year;
        }
    });

    const latestYear = Math.max(...data.map(r => parseInt(r.year) || 0));

    // Extract the administrative region names contained in the selected GeoJSON file.
    const regionNames = geojson.features.map(f => extractRegionName(f));

    // Match the predicted MPI values to the corresponding GeoJSON regions.
    const mpiValues = regionNames.map(name => {
        const normName = normaliseRegionName(name);

        // Use the prediction directly when the normalised regional names match.
        if (mpiLookup[normName] !== undefined) return mpiLookup[normName];

        // Attempt a partial name match when an exact regional-name match is unavailable.
        const partialKey = Object.keys(mpiLookup).find(k =>
            k.includes(normName) || normName.includes(k)
        );
        return partialKey ? mpiLookup[partialKey] : null;
    });

    // Identify the collection year associated with each mapped regional prediction.
    const yearUsedPerRegion = regionNames.map(name => {
        const normName = normaliseRegionName(name);

        if (yearLookup[normName]) return yearLookup[normName];

        const partialKey = Object.keys(yearLookup).find(k =>
            k.includes(normName) || normName.includes(k)
        );
        return partialKey ? yearLookup[partialKey] : null;
    });

    // Separate regions with predictions from regions for which no prediction is available.
    const predictedRegions = [];   // has MPI value → coloured
    const missingRegions   = [];   // no MPI value  → grey

    regionNames.forEach((name, i) => {
        if (mpiValues[i] !== null) {
            predictedRegions.push({
                name,
                mpi:     mpiValues[i],
                yearUsed: yearUsedPerRegion[i] || latestYear
            });
        } else {
            missingRegions.push({ name });
        }
    });

    console.log('Predicted regions:', predictedRegions.map(r => r.name));
    console.log('Missing regions (grey):', missingRegions.map(r => r.name));

    // Determine the GeoJSON property that Plotly should use to identify each region.
    const firstProps   = geojson.features[0].properties;
    const featureidkey = firstProps.shapeName ? 'properties.shapeName' :
                         firstProps.NAME_1    ? 'properties.NAME_1'    :
                         firstProps.name      ? 'properties.name'      :
                                                'properties.NAME_1';

    // Create the background map layer for regions without a predicted MPI value.
    // This layer is drawn first so regions with predictions can be displayed above it.
    const greyTrace = {
        type:        'choroplethmapbox',
        geojson:      geojson,
        locations:    missingRegions.map(r => r.name),
        featureidkey: featureidkey,
        z:            missingRegions.map(() => 0),   // dummy z value
        // colorscale:  [[0, '#cccccc'], [1, '#cccccc']], // solid grey
        colorscale:  [[0, '#707070'], [1, '#707070']], // darker grey
        zmin: 0,
        zmax: 1,
        showscale:   false,                           // hide colour bar for grey
        hovertemplate: missingRegions.map(r =>
            `<b>${r.name}</b><br>No prediction data<br>` +
            `<i>Region not selected for prediction</i><extra></extra>`
        ),
        text:          missingRegions.map(r => r.name),
        marker: { line: { width: 0.5, color: 'white' } },
        name: 'No prediction data'
    };

    // Create the coloured choropleth layer for regions with predicted MPI values.
    const hoverText = predictedRegions.map(r => {
        const status = r.mpi >= 0.3333
            ? '🔴 Poor (MPI ≥ 0.3333)'
            : '🟡 Non-Poor (MPI < 0.3333)';
        return `<b>${r.name}</b><br>` +
               `Predicted MPI: ${r.mpi.toFixed(3)}<br>` +
               `Year: ${r.yearUsed}<br>` +
               `${status}`;
    });

    const coloredTrace = {
        type:         'choroplethmapbox',
        geojson:       geojson,
        locations:     predictedRegions.map(r => r.name),
        featureidkey:  featureidkey,
        z:             predictedRegions.map(r => r.mpi),
        colorscale: [
            [0,      '#EF9F27'],   // yellow — non poor
            [0.3333, '#EF9F27'],
            [0.3334, '#E24B4A'],   // red — poor
            [1,      '#E24B4A']
        ],
        zmin:      0,
        zmax:      1,
        showscale: true,
        colorbar: {
            title:     'MPI Score',
            thickness: 15,
            tickvals:  [0, 0.3333, 1],
            ticktext:  ['0 (Non-Poor)', '0.3333 (threshold)', '1 (Poor)']
        },
        text:          hoverText,
        hovertemplate: '%{text}<extra></extra>',
        marker: { line: { width: 0.5, color: 'white' } },
        name: 'Predicted MPI'
    };


    // Add a separate legend entry for regions without predictions because the
    // choropleth layer does not provide the required custom legend entry directly.
    const greyLegendTrace = {
        type:  'scattermapbox',
        lat:   [null],        // null keeps it off the map entirely
        lon:   [null],
        mode:  'markers',
        marker: {
            size:  20,
            color: '#707070',
            opacity: 1
        },
        name:       'No prediction data — region not selected',
        showlegend:  true,
        hoverinfo:  'none'   // no hover tooltip for the dummy point
    };

    // Combine the available map layers before rendering the Plotly map.
    const traces = [];

    if (missingRegions.length > 0) traces.push(greyTrace);
    if (predictedRegions.length > 0) traces.push(coloredTrace);
    if (missingRegions.length > 0) traces.push(greyLegendTrace);

    // Calculate a suitable map centre from the selected GeoJSON boundaries.
    const allCoords = geojson.features.flatMap(f => {
        if (f.geometry.type === 'Polygon')
            return f.geometry.coordinates[0];
        if (f.geometry.type === 'MultiPolygon')
            return f.geometry.coordinates.flatMap(p => p[0]);
        return [];
    });
    const lats = allCoords.map(c => c[1]).filter(isFinite);
    const lons = allCoords.map(c => c[0]).filter(isFinite);

    // Define the choropleth map layout, legend and viewing area.
    const layout = {
        mapbox: {
            // style:  'carto-positron',
            style: 'open-street-map',
            center: {
                lat: (Math.min(...lats) + Math.max(...lats)) / 2,
                lon: (Math.min(...lons) + Math.max(...lons)) / 2
            },
            zoom: 5
        },
        margin: { t: 40, b: 0, l: 0, r: 0 },
        title:  `Predicted MPI — ${selectedcountry} (showing ${latestYear})`,
        showlegend: true,
        legend: {
            x:           0.01,
            y:           0.15,
            xanchor:     'left',
            yanchor:     'bottom',
            bgcolor:     'rgba(255,255,255,0.85)',
            bordercolor: '#cccccc',
            borderwidth: 1,
            font: { size: 12 }
        }
        /*
        legend: {
            x:           0.01,
            y:           0.15,
            xanchor:     'left',
            yanchor:     'bottom',
            bgcolor:     'rgba(255,255,255,0.85)',
            bordercolor: '#cccccc',
            borderwidth: 1,
            font: { size: 12 }
        }
        */
    };

    Plotly.newPlot('mapContainer', traces, layout, { responsive: true });
}

// Re-render the choropleth map when the user selects a different collection year.
function updateMapYear() {
    const selectedYear = parseInt(document.getElementById('mapYearSelect').value);
    const data = window.lastPredictions;

    if (!data) return;

    // Keep only predictions associated with the collection year selected for the map.
    const filteredData = data.filter(r => parseInt(r.year) === selectedYear);
    console.log(`Updating map for year ${selectedYear}:`, filteredData);

    // Retrieve the country code required for the selected map.
    const countryCodeMap = {
        'Ghana': 'GHA', 'Nigeria': 'NGA', 'Senegal': 'SEN',
        'Benin': 'BEN', 'Burkina Faso': 'BFA', 'Mali': 'MLI',
        'Niger': 'NER', 'Togo': 'TGO', 'Ivory Coast': 'CIV',
        'Guinea': 'GIN', 'Sierra Leone': 'SLE', 'Liberia': 'LBR',
        'Gambia': 'GMB', 'Mauritania': 'MRT'
    };

    //const countrycode = countryCodeMap[selectedcountry];
    const countrycode = getCountryCode(selectedcountry);
    if (countrycode) renderChoroplethMap(filteredData, countrycode);
}


// Control navigation between the different result visualisation tabs.
function switchTab(tabId) {
    // Hide all chart tabs before displaying the selected visualisation.
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.style.display = 'none';
    });

    // Remove the active state from each tab button.
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Display the chart associated with the selected tab.
    document.getElementById(tabId).style.display = 'block';

    // Mark the selected tab button as active.
    document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');

    // Resize Plotly after a previously hidden chart tab becomes visible.
    const containers = {
        barTab:  'barChartContainer',
        lineTab: 'lineChartContainer',
        mapTab:  'mapContainer'
    };

    const containerId = containers[tabId];
    if (containerId) {
        const el = document.getElementById(containerId);
        if (el && el.data) Plotly.relayout(el, { autosize: true });
    }

    // Resize the Chart.js classification chart when its tab becomes visible.
    if (tabId === 'donutTab' && window.donutChartInstance) {
        window.donutChartInstance.resize();
    }
}



// ============================================================================
// PAGE INITIALISATION AND USER INTERACTION
// ============================================================================
// Initialise the country selector and register the event handlers used for
// country, region, year, prediction and visualisation selections.
document.addEventListener('DOMContentLoaded', async () => {
    const dropdown = document.getElementById("countryDropdown");
    //let selectedcountry = '';

    try {
        // Retrieve the available countries from the Node.js backend when the page loads.
        const response = await fetch(`${NODE_API_URL}/api/countries`)

        
        if (!response.ok){
            throw new Error(`HTTP error! Status: ${response.status}`)
        }

        const countries = await response.json();
        //console.log(`countries variable: ${countries}`)
        console.log('countries:', JSON.stringify(countries, null, 2));

        // Load the country names and codes used by the remaining page functions.
        await loadCountryShortcodes();

        // Remove the temporary loading option before displaying the returned countries.
        dropdown.innerHTML = "";

        // Add a blank option so the user must make a country selection.
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = '-- Select a country --';
        dropdown.appendChild(defaultOption);

        countries.forEach(country => {
            const option = document.createElement('option');
            option.value = country.country_name;
            option.textContent = country.country_name;
            dropdown.appendChild(option);
        })

        // Update the regional controls when the selected country changes.
        dropdown.addEventListener('change', async (e) => {
            selectedcountry = e.target.value;

            // Clear previous selections and results before loading data for the new country.
            document.getElementById('dateSection').style.display = 'none';
            document.getElementById('dateList').innerHTML = '';
            document.getElementById('tableSection').style.display = 'none';
            document.getElementById('tableContainer').innerHTML = '';

            try {

                const regionalresponse = await fetch(`${NODE_API_URL}/api/regions?country=${encodeURIComponent(selectedcountry)}`)

                if (!regionalresponse.ok) {
                    throw new Error(`HTTP error! Status: ${regionalresponse.status}`);
                }

                const regions = await regionalresponse.json();
                console.table(regions); 
                renderRegionCheckboxes(regions); // ← renders checkboxes dynamically 

            } catch (error) {
                console.error('Error loading regions:', error)
            }


        });

        // Process the selected country, regions and years when the user submits the request.
        document.getElementById('submitBtn').addEventListener('click', async () => {
            const checked = [...document.querySelectorAll('#checkboxList input:checked')]
            .map(c => c.value);

            // Read the collection years selected by the user.
            const selectedYears = [...document.querySelectorAll('#dateList input:checked')]
            .map(c => c.value);

            if (!selectedcountry) {
                alert('Please select a country first.');
                return;
            }

            if (checked.length === 0) {
                alert('Please select at least one region.');
                return;
            }


            if (selectedYears.length === 0) {
                alert('Please select at least one year.');
                return;
            }

            console.log('Submitting:', { selectedcountry, checked, selectedYears });

            // Request MPI predictions for each selected year and combine the returned records.
            try {
                const allPredictions = [];

                for (const year of selectedYears) {
                    const yearDate = `${year}-01-01`;
                    const predictions = await fetchMPIPrediction(
                        selectedcountry,
                        checked,
                        yearDate
                    );

                    if (predictions) {
                        predictions.forEach(p => p.year = parseInt(year));
                        allPredictions.push(...predictions);
                    }
                }

                console.log('All predictions:', allPredictions);

                // Store the combined predictions so they can be reused by the visualisation controls.
                window.lastPredictions = allPredictions;

                // Display the combined MPI prediction results in the results table.
                renderTable(allPredictions);

                // render donut chart
                //renderDonutChart(allPredictions);   // ← add this line

                // Display the divider separating the prediction table from the visualisation controls.
                document.getElementById('dividerLine').style.display = 'block';

                // Display the prediction table and the available visualisation controls.
                document.getElementById('tableSection').style.display = 'block';
                document.getElementById('plotControlsSection').style.display = 'block';

                // Keep the chart area hidden until the user requests a visualisation.
                document.getElementById('chartSection').style.display = 'none';

            } catch (error) {
                console.error('Error generating predictions:', error);
            }

        });


        // Render the visualisation selected by the user when the Plot button is clicked.
        document.getElementById('plotBtn').addEventListener('click', async () => {
            const selectedChart = document.querySelector('input[name="chartType"]:checked').value;
            console.log('selected chart type:', selectedChart);

            const data = window.lastPredictions;

            if (!data || data.length === 0) {
                alert('No prediction data available to plot.');
                return;
            }

            // Prepare the country-code lookup required by the selected visualisation.
            const countryCodeMap = {
                'Ghana':        'GHA',
                'Nigeria':      'NGA',
                'Senegal':      'SEN',
                'Benin':        'BEN',
                'Burkina Faso': 'BFA',
                'Mali':         'MLI',
                'Niger':        'NER',
                'Togo':         'TGO',
                'Ivory Coast':  'CIV',
                'Guinea':       'GIN',
                'Sierra Leone': 'SLE',
                'Liberia':      'LBR',
                'Gambia':       'GMB',
                'Mauritania':   'MRT'
            };



            // Display the chart section after the visualisation has been prepared. for all chart types
            document.getElementById('chartSection').style.display = 'block';

            if (selectedChart === 'bar') {
                renderBarChart(data);
                switchTab('barTab');
                document.getElementById('chartSection').scrollIntoView({ behavior: 'smooth' });

            } else if (selectedChart === 'line') {
                renderLineChart(data);
                switchTab('lineTab');
                document.getElementById('chartSection').scrollIntoView({ behavior: 'smooth' });

            } else if (selectedChart === 'scatter') {
                renderScatterChart(data);
                switchTab('scatterTab');
                document.getElementById('chartSection').scrollIntoView({ behavior: 'smooth' });

            } else if (selectedChart === 'map') {
                //const countrycode = countryCodeMap[selectedcountry];
                const countrycode = getCountryCode(selectedcountry);
                console.log('countrycode: ' + countrycode);
                if (!countrycode) {
                    alert(`Country code not found for: ${selectedcountry}`);
                    return;
                }

                // Identify the collection years available for the choropleth map.
                const years = [...new Set(data.map(r => parseInt(r.year)))].sort();
                const yearSelector = document.getElementById('mapYearSelector');
                const yearSelect   = document.getElementById('mapYearSelect');

                switchTab('mapTab');

                if (years.length > 1) {
                    // Populate and display the map-year selector when multiple years are available.
                    yearSelect.innerHTML = years.map(y =>
                        `<option value="${y}" ${y === Math.max(...years) ? 'selected' : ''}>${y}</option>`
                        ).join('');
                    yearSelector.style.display = 'block';

                    // Display the most recent selected year when the map is first rendered.
                    const latestYear     = Math.max(...years);
                    const latestYearData = data.filter(r => parseInt(r.year) === latestYear);
                    await renderChoroplethMap(latestYearData, countrycode);

                } else {
                    // Hide the map-year selector when only one collection year is available.
                    if (yearSelector) yearSelector.style.display = 'none';
                    await renderChoroplethMap(data, countrycode);
                }

                document.getElementById('chartSection').scrollIntoView({ behavior: 'smooth' });
            } else if (selectedChart === 'donut') {
                renderDonutChart(data);
                switchTab('donutTab');
                document.getElementById('chartSection').scrollIntoView({ behavior: 'smooth' });
            }
        });

    } catch (error) {
        console.error(error)
        
    }

});