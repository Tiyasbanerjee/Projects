// Function to append numbers or operators to the display
function append_val(x){
    document.getElementById("display").value += x; // use "display", not "result"
}

// Function to clear the display
function clear_val(){
    document.getElementById("display").value = "";
}

// Function to calculate the expression
function calculate(){
    try {
        let x = document.getElementById("display").value;
        let y = eval(x);  // evaluate the expression
        document.getElementById("display").value = y; // show result
    } catch(err) {
        document.getElementById("display").value = "Error"; // show error if invalid
    }
}
