// Side bar collapse-expand
window.onload = function(){
    const sidebar = document.querySelector(".sidebar");
    const closeBtn = document.querySelector("#btn");
    const searchBtn = document.querySelector(".bx-search")

    closeBtn.addEventListener("click",function(){
        sidebar.classList.toggle("open")
        menuBtnChange()
    })

    searchBtn.addEventListener("click",function(){
        sidebar.classList.toggle("open")
        menuBtnChange()
    })

    function menuBtnChange(){
        if(sidebar.classList.contains("open")){
            closeBtn.classList.replace("bx-menu","bx-menu-alt-right")
        }else{
            closeBtn.classList.replace("bx-menu-alt-right","bx-menu")
        }
    }
}



///////////////////////
// For the search functionality
function handleSearch(event) {
    if (event.key === "Enter") {
        const searchInput = document.getElementById('searchProjectInput');
        const projectID = searchInput.value.trim();

        // Check if the input length is 14
        if (projectID.length !== 14) {
            alert('The Project ID must be exactly 14 characters long.');
            clearSearchInput(searchInput); // Clear the search input
            return;
        }

        // Check if the input is alphanumeric
        const alphanumericRegex = /^[a-zA-Z0-9]+$/;
        if (!alphanumericRegex.test(projectID)) {
            alert('The Project ID must contain only letters and numbers.');
            clearSearchInput(searchInput); // Clear the search input
            return;
        }

        // Send a GET request to the backend
        fetch(`/input_project/search/${projectID}/`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Project not found or expired.');
                }
                return response.json();
            })
            .then(data => {
                if (data.status === 'success') {
                    alert(`Project ID: ${data.project_id}\nPath: ${data.project_dir}`);
                } else {
                    alert(data.message || 'Project not found or expired.');
                    clearSearchInput(searchInput); // Clear the search input on failure
                }
            })
            .catch(error => {
                alert(error.message || 'An error occurred.');
                clearSearchInput(searchInput); // Clear the search input on failure
            });
    }
}

// Function to clear the search input after a delay
function clearSearchInput(inputElement) {
    inputElement.value = ''; // Clear the input field
}