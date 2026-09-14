// ============================================================================
// MULTIDIMENSIONAL POVERTY INDEX WEB APPLICATION - SINGLE PREDICTION PAGE
// ============================================================================
// This script provides the client-side functions used by the single-observation
// MPI prediction page. It retrieves the predictor variables expected by the
// deployed R Plumber API, builds the input form dynamically and allows the user
// to enter one set of predictor values for prediction.
//
// The script also checks whether manually entered values fall outside the range
// observed during model development, sends the completed observation to the
// prediction endpoint and displays the predicted MPI value, poverty
// classification and gauge visualisation returned by the API.
// ============================================================================

const PLUMBER_API_URL = window.APP_CONFIG?.PLUMBER_API_URL;


if (!PLUMBER_API_URL) {
    throw new Error("PLUMBER_API_URL is not configured");
}

console.log("Plumber API:", PLUMBER_API_URL);

// Display the current year in the copyright section of the page.
const copyRight = document.querySelector('.copyrightyear');
if (copyRight) {
    copyRight.innerText = '© ' + new Date().getFullYear() + ' Copyright';
}

// Store the predictor-variable information returned by the API so that it can
// be reused by the form, validation and prediction functions.
let predictorVariables = [];

// Retrieve the predictor variables and their summary information from the
// R Plumber API. These values are used to construct the manual input form.
async function fetchPredictorVariables() {
    try {
        const response = await fetch(`${PLUMBER_API_URL}/api/predictor_variables`);
        if (!response.ok) throw new Error(`HTTP error: ${response.status}`);

        const result = await response.json();
        console.log('Predictor variables raw:', result);

        // Handle the response status even if it is returned in a wrapped JSON structure.
        const status    = Array.isArray(result.status)    ? result.status[0]    : result.status;
        const variables = Array.isArray(result.variables) ? result.variables     : [result.variables];

        console.log('Status:', status);

        if (status !== 'success') {
            const message = Array.isArray(result.message) ? result.message[0] : result.message;
            throw new Error(message || 'Failed to fetch variables');
        }

        // Convert the returned variable properties to standard JavaScript values so
        // they can be used consistently when building the input controls.
        const unwrapped = variables.map(v => ({
            name:   Array.isArray(v.name)   ? String(v.name[0])   : String(v.name),
            type:   Array.isArray(v.type)   ? String(v.type[0])   : String(v.type),
            min:    Array.isArray(v.min)    ? Number(v.min[0])    : Number(v.min),
            max:    Array.isArray(v.max)    ? Number(v.max[0])    : Number(v.max),
            mean:   Array.isArray(v.mean)   ? Number(v.mean[0])   : Number(v.mean),
            median: Array.isArray(v.median) ? Number(v.median[0]) : Number(v.median)
        }));

        console.log('Variables sample after unwrap:', unwrapped[0]);
        console.log(`Total variables: ${unwrapped.length}`);

        return unwrapped;

    } catch (error) {
        console.error('Error fetching predictor variables:', error);
        return null;
    }
}

// Return a suitable display unit for each predictor based on the variable
// naming used in the dataset.
function getUnit(name) {
    if (name === 'ntlmeanintensity')   return 'nW/cm²/sr';  // nanowatts per cm² per steradian
    if (name === 'ntlcoverageperc')    return '% of pixels';
    if (name.startsWith('built'))      return 'count';
    if (name.endsWith('perc'))         return '% of pixels';           // sliders — already shown
    return '';
}

// Compare an entered value with the range observed in the model-development
// data and visually warn the user when the value falls outside that range.
function checkRange(input, min, max) {
    const val     = parseFloat(input.value);
    const warning = document.getElementById(`${input.id}_warning`);
    const hint    = document.getElementById(`${input.id}_hint`);

    if (!warning || !hint) return;

    if (!isNaN(val) && (val < min || val > max)) {
        // Mark the field when the entered value falls outside the observed training range.
        input.style.borderColor  = '#EF9F27';   // amber border
        warning.style.display    = 'block';
        hint.style.display       = 'none';
    } else {
        // Return the field to its normal appearance when the value is within range.
        input.style.borderColor  = '#cccccc';
        warning.style.display    = 'none';
        hint.style.display       = 'block';
    }
}

// Build the prediction form dynamically from the predictor information
// returned by the API rather than defining every input field manually.
function buildForm(variables) {
    const formContainer = document.getElementById('dynamicFormFields');
    if (!formContainer) return;

    formContainer.innerHTML = '';

    // Divide the predictors into the three groups used by the interface:
    // nighttime-light variables, land-cover variables and infrastructure variables.
    const ntlVars   = variables.filter(v => v.name.startsWith('ntl'));
    const percVars  = variables.filter(v => v.name.endsWith('perc') && !v.name.startsWith('ntl'));
    const builtVars = variables.filter(v => v.name.startsWith('built'));

    // Convert the stored predictor name into a more readable label for the web form.
    function formatLabel(name) {
        return name
            .replace(/^lcu/, '')           // remove lcu prefix
            .replace(/_/g, ' ')            // underscores to spaces
            .replace(/perc$/, '')          // remove perc suffix (slider shows % already)
            .replace(/\b\w/g, c => c.toUpperCase())  // title case
            .trim();
    }

    // Create a numeric input field for predictors that are entered directly as
    // continuous values or counts.
    function buildNumberField(v) {
        const label = formatLabel(v.name);
        const unit  = getUnit(v.name);
        const group = document.createElement('div');
        group.className = 'form-group';
        group.innerHTML = `
            <label class="form-variables" for="${v.name}">
                ${label}
                ${unit ? `<span class="unit-badge">${unit}</span>` : ''}
            </label>
            <input
                type="number"
                id="${v.name}"
                name="${v.name}"
                step="any"
                min="${v.min}"
                max="${v.max}"
                placeholder="e.g. ${v.median}"
                title="Min: ${v.min} | Max: ${v.max} | Mean: ${v.mean}"
                oninput="checkRange(this, ${v.min}, ${v.max})"
            />
            <span class="form-hint" id="${v.name}_hint">
                Range: ${v.min} – ${v.max} | Mean: ${v.mean}
            </span>
            <span class="range-warning" id="${v.name}_warning" style="display:none;">
                Value outside training range (${v.min} – ${v.max})
            </span>
    `;
    return group;
    }


    // Retain the earlier numeric-input implementation used during development.
// The active form-building function above is used for the current interface.
function buildNumberFieldold(v) {
        const label = formatLabel(v.name);
        const group = document.createElement('div');
        group.className = 'form-group';
        group.innerHTML = `
            <label class="form-variables" for="${v.name}">${label}</label>
            <input
                type="number"
                id="${v.name}"
                name="${v.name}"
                step="any"
                min="${v.min}"
                max="${v.max}"
                placeholder="e.g. ${v.median}"
                title="Min: ${v.min} | Max: ${v.max} | Mean: ${v.mean}"
            />
            <span class="form-hint">
                Range: ${v.min} – ${v.max} | Mean: ${v.mean}
            </span>
        `;
        return group;
    }

    // Create a slider input for land-cover predictors represented as percentages.
    function buildSliderField(v) {
        const label = formatLabel(v.name);

        // The displayed land-cover values use the percentage scale. Restrict the
        // slider display to a maximum of 100 percent.
        const sliderMin    = 0;
        const sliderMax    = 100;
        const sliderMedian = Math.min(Math.round(v.median), 100);
        const meanDisplay  = v.mean.toFixed(2);

        const group = document.createElement('div');
        group.className = 'form-group';
        group.innerHTML = `
            <label for="${v.name}">
                ${label}
                <span class="unit-badge">${getUnit(v.name)}</span>
            </label>
            <div class="slider-row">
                <input
                    type="range"
                    id="${v.name}"
                    name="${v.name}"
                    min="${sliderMin}"
                    max="${sliderMax}"
                    step="0.1"
                    value="${sliderMedian}"
                    oninput="document.getElementById('${v.name}_display').textContent = parseFloat(this.value).toFixed(1) + '%'"
                />
                <span class="slider-value" id="${v.name}_display">${sliderMedian}%</span>
            </div>
            <span class="form-hint">
                Data range: ${v.min}% – ${v.max}% | Mean: ${meanDisplay}%
            </span>
        `;
        return group;
    }

    // Create one predictor section and add the appropriate input controls to its
    // grid layout.
    function buildSection(title, vars, useSlider = false) {
        if (vars.length === 0) return;

        const section = document.createElement('div');
        section.className = 'form-section';

        const heading = document.createElement('p');
        heading.className = 'region-title';
        heading.textContent = title;
        section.appendChild(heading);

        const grid = document.createElement('div');
        grid.className = 'form-grid';

        vars.forEach(v => {
            const field = useSlider ? buildSliderField(v) : buildNumberField(v);
            grid.appendChild(field);
        });

        section.appendChild(grid);
        formContainer.appendChild(section);
    }

    // Add the nighttime-light, land-cover and infrastructure sections to the form.
    buildSection('Nighttime Light Variables', ntlVars, false);
    buildSection('Land Cover Variables', percVars, true);   // sliders
    buildSection('Built Infrastructure Variables', builtVars, false);
}

// Construct the query string that will be sent to the single-prediction API
// using the values currently entered in the form.
function buildQueryString() {
    return predictorVariables.map(v => {
        const el  = document.getElementById(v.name);
        let val   = el ? el.value.trim() : '0';

        // Send the entered value in the same scale used by the current API input.
        return `${v.name}=${encodeURIComponent(val)}`;
    }).join('&');
}


// Populate the form with the mean value recorded for each predictor so the
// user can start from a representative set of inputs.
function fillDefaults() {
    predictorVariables.forEach(v => {
        const el = document.getElementById(v.name);
        if (!el) return;

        if (v.name.endsWith('perc') && !v.name.startsWith('ntl')) {
            // Use the predictor mean for slider inputs while respecting the 100 percent display limit.
            const val = Math.min(parseFloat(v.mean.toFixed(1)), 100);
            el.value = val;

            // Update the percentage value displayed beside the slider.
            const display = document.getElementById(`${v.name}_display`);
            if (display) display.textContent = val + '%';
        } else {
            // Use the predictor mean as the default for numeric input fields.
            el.value = v.mean;
        }
    });
}

// Check that all required manual-input fields contain valid numeric values
// before the prediction request is sent.
function validateForm() {
    for (const v of predictorVariables) {
        const el  = document.getElementById(v.name);
        const val = el ? el.value.trim() : '';

        // Slider controls always contain a value, therefore validate the numeric fields separately.
        const isSlider = v.name.endsWith('perc') && !v.name.startsWith('ntl');

        if (!isSlider && (val === '' || isNaN(Number(val)))) {
            if (el) {
                el.style.borderColor = 'red';
                el.focus();
            }
            alert(`Please enter a valid number for: ${v.name}`);
            return false;
        }

        if (el && !isSlider) el.style.borderColor = '#ccc';
    }
    return true;
}


// Display the predicted MPI value using a gauge chart together with the
// study poverty threshold of 0.3333.
function renderGauge(mpiValue) {
    const trace = {
        type:  'indicator',
        mode:  'gauge+number+delta',
        value:  mpiValue,
        number: { valueformat: '.3f' },
        delta: {
            reference:  0.3333,
            increasing: { color: '#E24B4A' },
            decreasing: { color: '#1D9E75' }
        },
        gauge: {
            axis: {
                range:     [0, 1],
                tickvals:  [0, 0.1, 0.2, 0.3333, 0.5, 0.7, 1],
                ticktext:  ['0', '0.1', '0.2', '0.333', '0.5', '0.7', '1']
            },
            bar:   { color: mpiValue >= 0.3333 ? '#E24B4A' : '#EF9F27' },
            steps: [
                { range: [0, 0.3333], color: '#FFF8EC' },
                { range: [0.3333, 1], color: '#FFF0F0' }
            ],
            threshold: {
                line:      { color: 'red', width: 2 },
                thickness:  0.75,
                value:      0.3333
            }
        },
        title: { text: 'Predicted MPI (threshold: 0.3333)' }
    };

    Plotly.newPlot('gaugeChart', [trace], {
        margin: { t: 80, b: 20, l: 40, r: 40 },
        height: 300
    }, { responsive: true });

}

// Display the MPI prediction returned by the API together with the poverty
// classification, interpretation and any input-range warning.
function displayResult(result) {
    const mpi    = parseFloat(result.mpi_predicted);
    const isPoor = mpi >= 0.3333;

    document.getElementById('resultSection').style.display = 'block';
    document.getElementById('mpiScoreValue').textContent   = mpi.toFixed(3);

    document.getElementById('mpiScoreDisplay').style.background =
        isPoor ? '#FFF0F0' : '#F0FFF4';

    document.getElementById('mpiScoreValue').style.color =
        isPoor ? '#E24B4A' : '#1D9E75';

    const badge = document.getElementById('povertyStatusBadge');
    badge.textContent      = isPoor ? '🔴 Poor' : '🟡 Non-Poor';
    badge.style.background = isPoor ? '#FCEBEB' : '#FAEEDA';
    badge.style.color      = isPoor ? '#A32D2D' : '#854F0B';

    // Identify predictor values that were outside the observed training range so
    // the user can be informed that the prediction involves extrapolation.
    const outOfRange = predictorVariables.filter(v => {
        const el  = document.getElementById(v.name);
        const val = el ? parseFloat(el.value) : null;
        return val !== null && !isNaN(val) && (val < v.min || val > v.max);
    });

    // Build the explanatory text displayed below the predicted MPI result.
    let interpretation = result.interpretation;

    if (outOfRange.length > 0) {
        const names = outOfRange.map(v => formatVariableName(v.name)).join(', ');
        interpretation += ` Note: ${outOfRange.length} variable(s) were outside the training data range (${names}). Prediction accuracy may be reduced for these values.`;
        document.getElementById('interpretationText').style.color = '#A32D2D';
    }

    document.getElementById('interpretationText').textContent = interpretation;
    

    renderGauge(mpi);

    document.getElementById('resultSection')
        .scrollIntoView({ behavior: 'smooth' });
}

// Convert stored predictor names into readable labels for the range-warning message.
function formatVariableName(name) {
    return name
        .replace(/^lcu/, '')
        .replace(/_/g, ' ')
        .replace(/perc$/, '%')
        .replace(/\b\w/g, c => c.toUpperCase())
        .trim();
}

// ============================================================================
// USER INTERACTION AND PREDICTION REQUEST
// ============================================================================
// The following event handlers validate the form, submit the single observation
// to the R Plumber API, display the returned prediction and reset the interface
// when requested.
// Validate the completed form, send the predictor values to the R Plumber
// prediction endpoint and display the returned MPI result.
document.getElementById('predictBtn').addEventListener('click', async () => {
    if (!validateForm()) return;

    const predictBtn      = document.getElementById('predictBtn');
    predictBtn.textContent = 'Predicting...';
    predictBtn.disabled    = true;

    try {
        const queryString = buildQueryString();
        const url         = `${PLUMBER_API_URL}/api/predict_single?${queryString}`;
        console.log('Calling:', url);

        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP error: ${response.status}`);

        const result = await response.json();
        console.log('Result:', result);
        displayResult(result);

    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error calling prediction API. Make sure the R Plumber server is running on port 3796.');
    } finally {
        predictBtn.textContent = 'Predict MPI';
        predictBtn.disabled    = false;
    }
});

// Clear the manually entered predictor values and remove the current result.
document.getElementById('clearBtn').addEventListener('click', () => {
    predictorVariables.forEach(v => {
        const el = document.getElementById(v.name);
        if (!el) return;

        const isSlider = v.name.endsWith('perc') && !v.name.startsWith('ntl');

        if (isSlider) {
            // Reset slider inputs and update their displayed percentage values.
            el.value = 0;
            const display = document.getElementById(`${v.name}_display`);
            if (display) display.textContent = '0%';
        } else {
            // Clear the manually entered numeric value.
            el.value            = '';
            el.style.borderColor = '#ccc';
        }
    });

    // Hide the previous prediction result after the form is cleared.
    document.getElementById('resultSection').style.display = 'none';
});


// Fill the form with the predictor means returned by the API.
document.getElementById('defaultBtn').addEventListener('click', fillDefaults);

// ============================================================================
// PAGE INITIALISATION
// ============================================================================
// Retrieve the predictor information when the page loads and use it to construct
// the single-observation prediction form.
document.addEventListener('DOMContentLoaded', async () => {

    const formContainer  = document.getElementById('dynamicFormFields');
    const loadingMessage = document.getElementById('formLoadingMsg');

    // Display a temporary message while the predictor information is being retrieved.
    if (loadingMessage) loadingMessage.style.display = 'block';
    if (formContainer)  formContainer.innerHTML = '';

    // Retrieve the predictor information supplied by the deployed Plumber API.
    const variables = await fetchPredictorVariables();

    if (loadingMessage) loadingMessage.style.display = 'none';

    if (!variables || variables.length === 0) {
        if (formContainer) {
            formContainer.innerHTML = `
                <p style="color:red;">
                    Could not load predictor variables.
                    Make sure the R Plumber server is running on port 3796.
                </p>
            `;
        }
        return;
    }

    // Store the returned predictor information for later validation and display.
    predictorVariables = variables;
    console.log(`Loaded ${variables.length} predictor variables from recipe`);

    // Construct the manual prediction form from the returned predictor information.
    buildForm(variables);
});