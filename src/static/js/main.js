// Function to copy text to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        console.log('Async: Copying to clipboard was successful!');
    }, function(err) {
        console.error('Async: Could not copy text: ', err);
    });
}
// Custom JavaScript for Forex Pattern Discovery Framework

// Auto-hide flash messages after 5 seconds
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function() {
        const alerts = document.querySelectorAll('.alert');
        alerts.forEach(function(alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);

    // Copy error messages to clipboard when clicked
    document.querySelectorAll('.alert-danger').forEach(function(errorAlert) {
        errorAlert.addEventListener('click', function() {
            const errorMessage = errorAlert.textContent.trim();
            copyToClipboard(errorMessage);
            // Optionally, provide visual feedback to the user
            errorAlert.style.backgroundColor = '#d4edda'; // Light green
            errorAlert.style.borderColor = '#28a745';
            errorAlert.style.color = '#155724';
            errorAlert.textContent = 'Error copied to clipboard!';
            setTimeout(() => {
                // Revert to original state after a short delay
                errorAlert.style.backgroundColor = '';
                errorAlert.style.borderColor = '';
                errorAlert.style.color = '';
                errorAlert.textContent = errorMessage; // Restore original message
            }, 1500);
        });
    });
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
});
