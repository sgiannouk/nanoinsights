document.getElementById('find-project-tab').addEventListener('click', function() {
var searchBar = document.getElementById('search-bar');
if (searchBar.style.display === 'none' || searchBar.style.display === '') {
    searchBar.style.display = 'block';
} else {
    searchBar.style.display = 'none';
}
});

// Function to handle the search button click
function searchProject() {
// Get the project reference code from the input field
var projectReference = document.getElementById('project-reference').value;
// Perform the search or any other action here
console.log('Searching for project:', projectReference);
}