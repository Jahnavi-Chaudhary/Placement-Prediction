const form = document.getElementById('predictForm');
const resultBox = document.getElementById('result');
const probabilityValue = document.getElementById('probabilityValue');

form.addEventListener('submit', async function (event) {
  event.preventDefault();

  const studentData = {
    Age: parseInt(document.getElementById('age').value),
    Gender: document.getElementById('gender').value,
    CGPA: parseFloat(document.getElementById('cgpa').value),
    Internships: parseInt(document.getElementById('internships').value),
    Projects: parseInt(document.getElementById('projects').value),
    Coding_Skills: parseInt(document.getElementById('codingSkills').value),
    Communication_Skills: parseInt(document.getElementById('communicationSkills').value),
    Aptitude_Test_Score: parseInt(document.getElementById('aptitude').value),
    Soft_Skills_Rating: parseInt(document.getElementById('softSkills').value),
    Certifications: parseInt(document.getElementById('certifications').value),
    Backlogs: parseInt(document.getElementById('backlogs').value),
    Branch: document.getElementById('branch').value,
    Degree: document.getElementById('degree').value
  };

  try {
    const response = await fetch('https://placement-prediction-y29i.onrender.com/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(studentData)
    });

    const data = await response.json();
    const probabilityPercent = (data.probability * 100).toFixed(1);

    probabilityValue.textContent = probabilityPercent + '%';
    resultBox.classList.remove('hidden');
  } catch (error) {
    console.error('Prediction failed:', error);
    alert('Something went wrong. Check the console for details.');
  }
});