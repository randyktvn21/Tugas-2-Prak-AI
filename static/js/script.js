document.addEventListener('DOMContentLoaded', function () {
    // Form validation
    const form = document.querySelector('form');
    form.addEventListener('submit', function (e) {
        const luasTanah = document.querySelector('input[name="luas_tanah"]');
        if (luasTanah.value <= 0) {
            e.preventDefault();
            alert('Luas tanah harus lebih besar dari 0!');
        }
    });

    // Input animation
    const inputs = document.querySelectorAll('.form-control');
    inputs.forEach(input => {
        input.addEventListener('focus', function () {
            this.parentElement.classList.add('focused');
        });
        input.addEventListener('blur', function () {
            if (!this.value) {
                this.parentElement.classList.remove('focused');
            }
        });
    });
}); 