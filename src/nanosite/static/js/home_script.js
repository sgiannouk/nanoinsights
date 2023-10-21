// Get the validationType element
const validationType = document.getElementById("validationType");

// Get the extRawBox and extClinicalBox elements
const extRawBox = document.getElementById("extRawBox");
const extClinicalBox = document.getElementById("extClinicalBox");
const runBox = document.getElementById("runBox");

// Add an event listener to validationType
validationType.addEventListener("change", function () {
    // Get the selected option value
    const selectedOption = validationType.value;

    // Show or hide extRawBox, extClinicalBox and runBox based on the selected option
    if (selectedOption === "run") {
            runBox.style.display = "block";
            extRawBox.style.display = "none"; // Hide extRawBox
            extClinicalBox.style.display = "none"; // Hide extClinicalBox
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



////////////////////////////
// JavaScript to show/hide the "edgernormselection" section based on the "norm" dropdown
document.getElementById("norm").addEventListener("change", function() {
    var selectedValue = this.value;
    var edgernormSelection = document.getElementById("edgernormselection");

    if (selectedValue === "edger") {
        edgernormSelection.style.display = "block"; // Show the section
    } else {
        edgernormSelection.style.display = "none"; // Hide the section
    }
});



////////////////////////////
// Get references to the relevant elements
const featureSelectionDropdown = document.getElementById("featureselection");
const rfecrossvalField = document.getElementById("rfecrossvalopt");
const minfeatureField = document.getElementById("minfeatureopt");

// Function to show/hide the fields based on the selected value
function toggleFields() {
    if (featureSelectionDropdown.value === "RFE") {
        rfecrossvalField.style.display = "block";
        minfeatureField.style.display = "block";
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
//Submit button
$(function() {
    $( "#button" ).click(function() {
        $( "#button" ).addClass( "onclic", 250, validate);
    });

    function validate() {
        setTimeout(function() {
        $( "#button" ).removeClass( "onclic" );
        $( "#button" ).addClass( "validate", 450, callback );
        }, 2250 );
    }
        function callback() {
        setTimeout(function() {
            $( "#button" ).removeClass( "validate" );
        }, 1250 );
        }
});