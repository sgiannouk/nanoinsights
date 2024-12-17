document.addEventListener('DOMContentLoaded', function () {
    // Select the sidebar and the burger button
    const sidebar = document.querySelector('.sidebar');
    const burgerButton = document.getElementById('btn');

    // Add a click event listener to toggle the "open" class
    burgerButton.addEventListener('click', function () {
        sidebar.classList.toggle('open'); // Toggle the "open" class
    });
});



////////////////////////////
// Event listener for training data uploads
function handleFileUpload(files, fieldName, uploadType = 'training') {
    const formData = new FormData();

    // Map variable names to user-friendly descriptions
    const variableAnnotations = {
        "raw_files[]": "Raw Data Files",
        "clinical_file": "Clinical Data File",
        "ext_raw_files[]": "External Raw Data Files",
        "ext_clinical_file": "External Clinical Data File"
    };

    // Resolve the user-friendly name for the field or fall back to the variable name
    const friendlyName = variableAnnotations[fieldName] || fieldName;

    // Add CSRF token
    formData.append('csrfmiddlewaretoken', getCsrfToken());

    // Append files to the FormData object
    if (fieldName.endsWith('raw_files[]')) {
        for (let i = 0; i < files.length; i++) {
            formData.append(fieldName, files[i]);
        }
    } else {
        formData.append(fieldName, files[0]); // Only one clinical file allowed
    }

    // Add upload type
    formData.append('upload_type', uploadType);

    // Send the files to the server
    fetch('/input_project/upload/', {
        method: 'POST',
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Success message with user-friendly annotation
                showNotification(`${friendlyName} uploaded successfully!`, true);
            } else {
                // Error message from the server
                showNotification(`Upload failed: ${data.message}`, false);
            }
        })
        .catch(error => {
            console.error('Error during file upload:', error);
            // Generic error message
            showNotification(`An error occurred while uploading ${friendlyName}.`, false);
        });
}

// Event Listeners for Training Data
document.getElementById('rawTrainingData').addEventListener('change', function () {
    handleFileUpload(this.files, 'raw_files[]', 'training');
});

document.getElementById('clinicalTrainingData').addEventListener('change', function () {
    handleFileUpload(this.files, 'clinical_file', 'training');
});

// Event Listeners for Validation Data
document.getElementById('rawValidationData').addEventListener('change', function () {
    handleFileUpload(this.files, 'ext_raw_files[]', 'validation');
});

document.getElementById('clinicalValidationData').addEventListener('change', function () {
    handleFileUpload(this.files, 'ext_clinical_file', 'validation');
});

function getCsrfToken() {
    const token = document.querySelector('[name=csrfmiddlewaretoken]');
    if (!token) {
        console.error("CSRF token not found in the DOM.");
        return null;
    }
    return token.value;
}




////////////////////////////
// Read the clinical data file and show the Control and Condition groups
document.getElementById('clinicalTrainingData').addEventListener('change', function () {
    const file = this.files[0];

    if (!file) {
        alert('No file selected.');
        return;
    }

    // Validate file type
    const validTypes = ['text/csv', 'text/plain'];
    if (!validTypes.includes(file.type)) {
        alert('Invalid file type. Please upload a CSV or TXT file.');
        return;
    }

    // Read the file
    const reader = new FileReader();
    reader.onload = function (event) {
        const fileContent = event.target.result;

        // Parse the file content
        const rows = fileContent.split('\n').map(row => row.trim());
        const header = rows[0].split(',');
        const conditionIndex = header.indexOf('Condition');

        if (conditionIndex === -1) {
            alert('The file must contain a "Condition" column.');
            return;
        }

        // Extract unique values from the "Condition" column
        const uniqueConditions = new Set();
        for (let i = 1; i < rows.length; i++) {
            const columns = rows[i].split(',');
            if (columns.length > conditionIndex) {
                uniqueConditions.add(columns[conditionIndex].trim());
            }
        }

        // Validate the number of unique conditions
        if (uniqueConditions.size !== 2) {
            alert('The "Condition" column must have exactly two unique values.');
            return;
        }

        // Populate the dropdowns
        const [condition1, condition2] = Array.from(uniqueConditions);
        const controlDropdown = document.getElementById('control');
        const conditionDropdown = document.getElementById('condition');

        // Clear existing options
        controlDropdown.innerHTML = `<option value="">Select Control</option>`;
        conditionDropdown.innerHTML = `<option value="">Select Condition</option>`;

        // Add new options
        controlDropdown.innerHTML += `
            <option value="${condition1}">${condition1}</option>
            <option value="${condition2}">${condition2}</option>
        `;
        conditionDropdown.innerHTML += `
            <option value="${condition1}">${condition1}</option>
            <option value="${condition2}">${condition2}</option>
        `;

        // Add event listeners to dynamically enable/disable options
        controlDropdown.addEventListener('change', function () {
            updateDropdownStates(controlDropdown, conditionDropdown, condition1, condition2);
        });

        conditionDropdown.addEventListener('change', function () {
            updateDropdownStates(conditionDropdown, controlDropdown, condition1, condition2);
        });
    };

    reader.onerror = function () {
        alert('Failed to read the file.');
    };

    reader.readAsText(file);
});

// Function to update dropdown states
function updateDropdownStates(changedDropdown, otherDropdown, condition1, condition2) {
    const selectedValue = changedDropdown.value;

    // Update the other dropdown's options
    Array.from(otherDropdown.options).forEach(option => {
        if (option.value === selectedValue) {
            option.disabled = true;
        } else {
            option.disabled = false;
        }
    });
}




////////////////////////////
// Get the testingType element
const testingType = document.getElementById("testingType");

// Get the extRawBox, extClinicalBox, and runBox elements
const extRawBox = document.getElementById("extRawBox");
const extClinicalBox = document.getElementById("extClinicalBox");
const runBox = document.getElementById("runBox");
const runSelect = document.getElementById("run");

// Add an event listener to testingType
testingType.addEventListener("change", function () {
    const selectedOption = testingType.value;
    const clinicalFileInput = document.getElementById("clinicalTrainingData");

    if (selectedOption === "run") {
        if (!clinicalFileInput.files.length) {
            showNotification("Please upload the NanoString Raw Data Files before selecting 'Runs' as the validation type.", false);
            testingType.value = "split"; // Reset to a default value
            return;
        }
        runBox.style.display = "block";
        extRawBox.style.display = "none"; // Hide extRawBox
        extClinicalBox.style.display = "none"; // Hide extClinicalBox

        // Fetch CartridgeIDs from the backend
        fetchCartridgeIDs();
    } else if (selectedOption === "extSet") {
        extRawBox.style.display = "block";
        extClinicalBox.style.display = "block";
        runBox.style.display = "none"; // Hide runBox
    } else {
        runBox.style.display = "none";
        extRawBox.style.display = "none";
        extClinicalBox.style.display = "none";
    }
});


// Function to fetch CartridgeIDs from the backend
function fetchCartridgeIDs() {
    const projectId = getProjectIdFromSession(); // Retrieve the project ID from the session or context
    if (!projectId) {
        alert("Project ID is missing. Ensure you have uploaded files.");
        return;
    }

    fetch(`/input_project/get_cartridge_ids/?project_id=${projectId}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === "success") {
                updateRunBox(new Set(data.cartridge_ids));
            } else {
                alert(`Error fetching CartridgeIDs: ${data.message}`);
            }
        })
        .catch(error => {
            console.error("Error fetching CartridgeIDs:", error);
            alert("An error occurred while fetching CartridgeIDs.");
        });
}

// Function to update the runBox dropdown
function updateRunBox(cartridgeIDs) {
    runSelect.innerHTML = ""; // Clear previous options

    cartridgeIDs.forEach(id => {
        const option = document.createElement("option");
        option.value = id;
        option.textContent = id;
        runSelect.appendChild(option);
    });
}

// Utility function to retrieve the project ID from the session
function getProjectIdFromSession() {
    const projectInput = document.getElementById("projectId");
    if (!projectInput || !projectInput.value) {
        alert("Project ID is missing. Ensure you have uploaded files.");
        return null;
    }
    return projectInput.value;
}




////////////////////////////
// Hide/Show rfecrossvalField and minfeatureField based on featureSelectionDropdown
const featureSelectionDropdown = document.getElementById("featureselection");
const rfecrossvalField = document.getElementById("rfecrossvalopt");
const minfeatureField = document.getElementById("minfeatureopt");

function toggleFields() {
    if (featureSelectionDropdown.value === "RFE") {
        rfecrossvalField.style.display = "block";
        minfeatureField.style.display = "block";
    } else if (featureSelectionDropdown.value === 'PI') {
        rfecrossvalField.style.display = "none";
    } else {
        rfecrossvalField.style.display = "none";
        minfeatureField.style.display = "none";
    }
}
// Add an event listener to the "featureselection" dropdown
featureSelectionDropdown.addEventListener("change", toggleFields);

// Initially, call the toggleFields function to set the initial state
toggleFields();



////////////////////////////
// Collapse the Advanced Options menus
document.getElementById("toggleAdvancedOptions").addEventListener("click", function() {
    var advancedOptions = document.getElementById("collapseadvancedoptions");
    if (advancedOptions.classList.contains("show")) {
        advancedOptions.classList.remove("show");
    } else {
        advancedOptions.classList.add("show");
    }
});

document.addEventListener("DOMContentLoaded", function () {
    var toggleButton = document.getElementById("toggleAdvancedOptions");
    var collapseElement = document.getElementById("collapseadvancedoptions");

    toggleButton.addEventListener("click", function () {
        if (collapseElement.classList.contains("show")) {
            // Collapse is shown; scroll to the bottom
            collapseElement.scrollIntoView({ behavior: "smooth" });
        } else {
            // Collapse is hidden; scroll to the top
            toggleButton.scrollIntoView({ behavior: "smooth" });
        }
    });
});



// Collapse the Filter Options menus
document.getElementById("toggleFilterOptions").addEventListener("click", function() {
    var filterOptions = document.getElementById("collapsefilteroptions");
    if (filterOptions.classList.contains("show")) {
        filterOptions.classList.remove("show");
    } else {
        filterOptions.classList.add("show");
    }
});

document.addEventListener("DOMContentLoaded", function () {
    var toggleButton = document.getElementById("toggleFilterOptions");
    var collapseElement = document.getElementById("collapsefilterdoptions");

    toggleButton.addEventListener("click", function () {
        if (collapseElement.classList.contains("show")) {
            // Collapse is shown; scroll to the bottom
            collapseElement.scrollIntoView({ behavior: "smooth" });
        } else {
            // Collapse is hidden; scroll to the top
            toggleButton.scrollIntoView({ behavior: "smooth" });
        }
    });
});



////////////////////////////
// Change the icon next to "Advanced Options" and "Filters" when you click on it
function toggleIcon() {
    const icon = document.getElementById('toggleIcon');
    if (icon.getAttribute('name') === 'chevron-down-outline') {
        icon.setAttribute('name', 'chevron-up-outline'); // Change to the desired icon name
    } else {
        icon.setAttribute('name', 'chevron-down-outline'); // Change back to the original icon
    }
}



////////////////////////////
// Submit button
document.getElementById('button').addEventListener('click', function (event) {
    event.preventDefault(); // Prevent default form submission

    const projectId = document.getElementById('projectId').value; // Retrieve the project ID from the hidden input
    const formData = new FormData();
    const missingFields = [];


    // Map variable names to user-friendly descriptions
    const variableAnnotations = {
        "raw_files[]": "Raw Data Files",
        "clinical_file": "Clinical Data File",
        "control": "Control Group",
        "condition": "Condition Group",
        "run[]": "Runs"
    };

    // Check required fields
    if (document.querySelector('input[name="raw_files[]"]').files.length === 0) {
        missingFields.push(variableAnnotations["raw_files[]"]);
    }
    if (document.querySelector('input[name="clinical_file"]').files.length === 0) {
        missingFields.push(variableAnnotations["clinical_file"]);
    }
    if (document.querySelector('select[name="control"]').value === '') {
        missingFields.push(variableAnnotations["control"]);
    }
    if (document.querySelector('select[name="condition"]').value === '') {
        missingFields.push(variableAnnotations["condition"]);
    }

    // Check for "Runs" validation if testingType is "run"
    const testingType = document.getElementById('testingType').value;
    if (testingType === 'run') {
        const selectedRuns = document.querySelector('select[name="run[]"]').selectedOptions;
        if (selectedRuns.length === 0) {
            missingFields.push("At least one run should be chosen");
        }
    }

    if (missingFields.length > 0) {
        const message = missingFields.length === 1
            ? `The following field is missing: ${missingFields[0]}.\nPlease complete it before submitting!`
            : `The following fields are missing: ${missingFields.join(", ")}.\nPlease complete them before submitting!`;

        // Show notification
        showNotification(message, false);
        return; // Stop further execution
    }

    // Collect multi-select inputs for 'run' and 'classifiers'
    document.querySelectorAll('select[multiple]').forEach(select => {
        Array.from(select.selectedOptions).forEach(option => {
            formData.append(select.name, option.value);
        });
    });

    // Collect all inputs from the left and right containers
    document.querySelectorAll('.container-left input, .container-left select, .container-right input, .container-right select').forEach(input => {
        if (input.type === 'file') {
            Array.from(input.files).forEach(file => {
                formData.append(input.name, file);
            });
        } else if (input.type === 'checkbox' || input.type === 'radio') {
            if (input.checked) {
                formData.append(input.name, input.value);
            }
        } else if (!formData.has(input.name)) {
            formData.append(input.name, input.value);
        }
    });

    // Add a submit indicator
    formData.append('submit', 'true');

    // Disable the submit button while validation is ongoing
    const submitButton = document.getElementById('button');
    submitButton.disabled = true;

    // Send data to the server for validation and submission
    fetch('/input_project/upload/', {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': getCsrfToken(), // Ensure CSRF token is included
        },
    })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showNotification('Files uploaded and configuration saved!', true);

                // Activate the "My Analysis" tab
                const myAnalysisLink = document.getElementById('myAnalysis');
                if (myAnalysisLink) {
                    myAnalysisLink.classList.remove('inactive'); // Remove inactive class
                    myAnalysisLink.style.pointerEvents = 'auto'; // Enable clicking
                    myAnalysisLink.style.color = ''; // Reset text color to default
                    myAnalysisLink.style.cursor = ''; // Reset cursor to default
                    myAnalysisLink.setAttribute('href', 'analysis.html'); // Set the link to analysis.html
                }

                // NEW: Trigger the nanoinsights_init.py execution
                return fetch(`/input_project/run-nanoinsights/${projectId}/`, { // Include project_id in the URL
                    method: 'POST',
                    headers: {
                        "X-CSRFToken": getCsrfToken(),
                    },
                });
            } else {
                // Show error notification for validation issues
                showNotification(data.message, false);
                throw new Error('Upload failed');
            }
        })
        .then(runResponse => runResponse.json())
        .then(runData => {
            if (runData && runData.success) {
                showNotification('NanoInsights script executed successfully!', true);
            } else if (runData && runData.error) {
                showNotification(`Error running NanoInsights: ${runData.error}`, false);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification('An error occurred while submitting the form or running NanoInsights.', false);
        })
        .finally(() => {
            // Re-enable the submit button after completion
            submitButton.disabled = false;
        });
});



// document.getElementById('button').addEventListener('click', function (event) {
//     event.preventDefault(); // Prevent default form submission

//     const formData = new FormData();
//     const missingFields = [];

//     // Map variable names to user-friendly descriptions
//     const variableAnnotations = {
//         "raw_files[]": "Raw Data Files",
//         "clinical_file": "Clinical Data File",
//         "control": "Control Group",
//         "condition": "Condition Group",
//         "run[]": "Runs"
//     };

//     // Check required fields
//     if (document.querySelector('input[name="raw_files[]"]').files.length === 0) {
//         missingFields.push(variableAnnotations["raw_files[]"]);
//     }
//     if (document.querySelector('input[name="clinical_file"]').files.length === 0) {
//         missingFields.push(variableAnnotations["clinical_file"]);
//     }
//     if (document.querySelector('select[name="control"]').value === '') {
//         missingFields.push(variableAnnotations["control"]);
//     }
//     if (document.querySelector('select[name="condition"]').value === '') {
//         missingFields.push(variableAnnotations["condition"]);
//     }

//     // Check for "Runs" validation if testingType is "run"
//     const testingType = document.getElementById('testingType').value;
//     if (testingType === 'run') {
//         const selectedRuns = document.querySelector('select[name="run[]"]').selectedOptions;
//         if (selectedRuns.length === 0) {
//             missingFields.push("At least one run should be chosen");
//         }
//     }

//     if (missingFields.length > 0) {
//         const message = missingFields.length === 1
//             ? `The following field is missing: ${missingFields[0]}.\nPlease complete it before submitting!`
//             : `The following fields are missing: ${missingFields.join(", ")}.\nPlease complete them before submitting!`;

//         // Show notification
//         showNotification(message, false);
//         return; // Stop further execution
//     }

//     // Collect multi-select inputs for 'run' and 'classifiers'
//     document.querySelectorAll('select[multiple]').forEach(select => {
//         Array.from(select.selectedOptions).forEach(option => {
//             formData.append(select.name, option.value);
//         });
//     });

//     // Collect all inputs from the left and right containers
//     document.querySelectorAll('.container-left input, .container-left select, .container-right input, .container-right select').forEach(input => {
//         if (input.type === 'file') {
//             Array.from(input.files).forEach(file => {
//                 formData.append(input.name, file);
//             });
//         } else if (input.type === 'checkbox' || input.type === 'radio') {
//             if (input.checked) {
//                 formData.append(input.name, input.value);
//             }
//         } else if (!formData.has(input.name)) {
//             formData.append(input.name, input.value);
//         }
//     });

//     // Add a submit indicator
//     formData.append('submit', 'true');

//     // Disable the submit button while validation is ongoing
//     const submitButton = document.getElementById('button');
//     submitButton.disabled = true;

//     // Send data to the server for validation and submission
//     fetch('/input_project/upload/', {
//         method: 'POST',
//         body: formData,
//         headers: {
//             'X-CSRFToken': getCsrfToken(), // Ensure CSRF token is included
//         },
//     })
//         .then(response => response.json())
//         .then(data => {
//             if (data.status === 'success') {
//                 showNotification('Files uploaded and configuration saved!', true);

//                 // Activate the "My Analysis" tab
//                 const myAnalysisLink = document.getElementById('myAnalysis');
//                 if (myAnalysisLink) {
//                     myAnalysisLink.classList.remove('inactive'); // Remove inactive class
//                     myAnalysisLink.style.pointerEvents = 'auto'; // Enable clicking
//                     myAnalysisLink.style.color = ''; // Reset text color to default
//                     myAnalysisLink.style.cursor = ''; // Reset cursor to default
//                     myAnalysisLink.setAttribute('href', 'analysis.html'); // Set the link to analysis.html
//                 }
//             } else {
//                 // Show error notification for validation issues
//                 showNotification(data.message, false);
//             }
//         })
//         .catch(error => {
//             console.error('Error submitting form:', error);
//             showNotification('An error occurred while submitting the form.', false);
//         })
//         .finally(() => {
//             // Re-enable the submit button after completion
//             submitButton.disabled = false;
//         });
// });

// // Ensure the "My Analysis" tab is inactive on page load
// document.addEventListener('DOMContentLoaded', function () {
//     const myAnalysisLink = document.getElementById('myAnalysis');
//     if (myAnalysisLink) {
//         myAnalysisLink.classList.add('inactive');
//         myAnalysisLink.style.pointerEvents = 'none'; // Disable clicks
//         myAnalysisLink.style.color = 'gray'; // Change text color to gray
//         myAnalysisLink.removeAttribute('href'); // Remove the href to prevent navigation
//     }
// });







////////////////////////////
// Helper function to show notifications
function showNotification(message, isSuccess) {
    const container = document.getElementById('notification-container');
    const notification = document.createElement('div');
    notification.className = `notification ${isSuccess ? 'success' : 'error'}`;
    notification.innerText = message;

    container.appendChild(notification);

    // Set display duration: 3 seconds for success, 6 seconds for error
    const displayDuration = isSuccess ? 3000 : 6000;

    setTimeout(() => {
        notification.classList.add('fadeOut');
        setTimeout(() => notification.remove(), 1000); // Wait for animation to finish
    }, displayDuration);
}

////////////////////////////
// Helper function to get CSRF token
function getCsrfToken() {
    const token = document.querySelector('[name=csrfmiddlewaretoken]');
    if (!token) {
        console.error("CSRF token not found in the DOM.");
        return null;
    }
    return token.value;
}





//////////////////////
window.addEventListener('beforeunload', function () {
    const projectId = document.getElementById('projectId')?.value;
    if (projectId) {
        const configPathExists = document.querySelector('#configExists')?.value === 'true';
        if (!configPathExists) {
            navigator.sendBeacon('/delete_project/', JSON.stringify({ project_id: projectId }));
        }
    }
});



//////////////////////
// Functions for Notifications
function showNotification(message, isSuccess) {
    const container = document.getElementById('notification-container');
    const notification = document.createElement('div');
    notification.className = `notification ${isSuccess ? 'success' : 'error'}`;
    notification.innerText = message;

    container.appendChild(notification);

    // Set display duration: 3 seconds for success, 8 seconds for error
    const displayDuration = isSuccess ? 3000 : 6000;

    setTimeout(() => {
        notification.classList.add('fadeOut');
        setTimeout(() => notification.remove(), 1000); // Wait for animation to finish
    }, displayDuration);
}
