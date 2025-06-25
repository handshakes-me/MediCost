// Initialize Materialize components
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all Materialize dropdowns
    var dropdowns = document.querySelectorAll('.dropdown-trigger');
    M.Dropdown.init(dropdowns, {
        coverTrigger: false
    });

    // Initialize form select
    var selects = document.querySelectorAll('select');
    M.FormSelect.init(selects);

    // Initialize modals
    var modals = document.querySelectorAll('.modal');
    M.Modal.init(modals);

    // Initialize floating action button
    var fab = document.querySelectorAll('.fixed-action-btn');
    M.FloatingActionButton.init(fab);
});
