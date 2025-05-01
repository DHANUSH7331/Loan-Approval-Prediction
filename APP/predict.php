<?php
$output = ""; // Initialize

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $annual_income = $_POST['annual_income'] ?? null;
    $loan_amount = $_POST['loan_amount'] ?? null;
    $credit_score = $_POST['credit_score'] ?? null;
    $employment_type = $_POST['employment_type'] ?? null;
    $loan_term = $_POST['loan_term'] ?? null;
    $existing_loans = $_POST['existing_loans'] ?? null;
    $model = $_POST['model'] ?? null;

    if (!$annual_income || !$loan_amount || !$credit_score || !$employment_type || !$loan_term || !$existing_loans || !$model) {
        $output = "All fields are required.";
    } else {
        $existing_loans = strtolower($existing_loans) === 'yes' ? 1 : 0;

        if ($model == 'logistic') {
            $model_choice = 1;
        } elseif ($model == 'random_forest') {
            $model_choice = 2;
        } elseif ($model == 'xgboost') {
            $model_choice = 3;
        } else {
            $output = "Invalid model selected.";
        }

        if (!isset($output) || $output == "") {
            $python = 'C:/Users/DHANUSH/AppData/Local/Programs/Python/Python313/python.exe';
            $script = 'predict.py';

            $command = "$python " . escapeshellarg($script) . 
                    " " . escapeshellarg($annual_income) . 
                    " " . escapeshellarg($loan_amount) . 
                    " " . escapeshellarg($credit_score) . 
                    " " . escapeshellarg($employment_type) . 
                    " " . escapeshellarg($loan_term) . 
                    " " . escapeshellarg($existing_loans) . 
                    " " . escapeshellarg($model_choice);

            $output = trim(shell_exec($command));
            $output_lines = explode("\n", $output); // Split output into result and confidence score
            $result = $output_lines[0] ?? 'No result'; // First line is the result (Approved/Rejected)
            $confidence = $output_lines[1] ?? 'No confidence score'; // Second line is the confidence score
        }
    }
} else {
    $output = "Invalid Request.";
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Prediction Result</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="form-container">
        <h2>Loan Prediction Result</h2>
        <div class="form-group">
            <h3>Prediction Result: <?php echo htmlspecialchars($result); ?></h3> <!-- Display Prediction Result -->
            <h3>Confidence Score: <?php echo htmlspecialchars($confidence); ?>%</h3> <!-- Display Confidence Score -->
        </div>
        <a href="loan_form.html" class="btn">Predict Again</a>
    </div>
</body>
</html>
