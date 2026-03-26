const SYMPTOMS = [
  "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering",
  "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting",
  "vomiting", "burning_micturition", "spotting_ urination", "fatigue", "weight_gain",
  "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness",
  "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", "high_fever",
  "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache",
  "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes",
  "back_pain", "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
  "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach",
  "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm",
  "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion",
  "chest_pain", "weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements",
  "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness",
  "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels",
  "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties",
  "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips", "slurred_speech",
  "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints",
  "movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness",
  "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort", "foul_smell_of urine",
  "continuous_feel_of_urine", "passage_of_gases", "internal_itching", "toxic_look_(typhos)",
  "depression", "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body",
  "belly_pain", "abnormal_menstruation", "dischromic _patches", "watering_from_eyes",
  "increased_appetite", "polyuria", "family_history", "mucoid_sputum", "rusty_sputum",
  "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion",
  "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen",
  "history_of_alcohol_consumption", "fluid_overload.1", "blood_in_sputum",
  "prominent_veins_on_calf", "palpitations", "painful_walking", "pus_filled_pimples",
  "blackheads", "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails",
  "inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze"
];

const CATEGORIES = {
  general: ["fatigue", "high_fever", "mild_fever", "chills", "shivering", "sweating", "weight_loss", "weight_gain", "malaise", "lethargy", "dehydration", "loss_of_appetite", "anxiety", "mood_swings", "restlessness", "depression", "irritability", "obesity", "family_history", "altered_sensorium", "coma"],
  skin: ["itching", "skin_rash", "nodal_skin_eruptions", "yellowish_skin", "dischromic _patches", "skin_peeling", "silver_like_dusting", "pus_filled_pimples", "blackheads", "scurring", "blister", "red_sore_around_nose", "yellow_crust_ooze", "red_spots_over_body", "bruising", "inflammatory_nails", "brittle_nails", "small_dents_in_nails"],
  digestive: ["stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", "nausea", "indigestion", "constipation", "abdominal_pain", "diarrhoea", "belly_pain", "passage_of_gases", "internal_itching", "loss_of_appetite", "swelling_of_stomach", "distention_of_abdomen", "stomach_bleeding", "bloody_stool", "pain_during_bowel_movements", "pain_in_anal_region", "irritation_in_anus", "acute_liver_failure", "fluid_overload"],
  respiratory: ["cough", "breathlessness", "phlegm", "throat_irritation", "runny_nose", "congestion", "sinus_pressure", "chest_pain", "mucoid_sputum", "rusty_sputum", "blood_in_sputum"],
  neuro: ["headache", "dizziness", "blurred_and_distorted_vision", "loss_of_balance", "unsteadiness", "spinning_movements", "slurred_speech", "weakness_of_one_body_side", "loss_of_smell", "lack_of_concentration", "visual_disturbances", "altered_sensorium", "coma", "stiff_neck", "movement_stiffness"],
  musculo: ["joint_pain", "back_pain", "neck_pain", "knee_pain", "hip_joint_pain", "muscle_weakness", "muscle_wasting", "muscle_pain", "swelling_joints", "weakness_in_limbs", "cramps", "painful_walking"],
  urinary: ["burning_micturition", "spotting_ urination", "yellow_urine", "dark_urine", "bladder_discomfort", "foul_smell_of urine", "continuous_feel_of_urine", "polyuria"]
};

const DISEASE_DESC = {
  "AIDS": "A serious condition caused by HIV that damages the immune system.",
  "Acne": "A skin condition causing pimples, blackheads and inflammation.",
  "Alcoholic Hepatitis": "Liver inflammation caused by excessive alcohol consumption.",
  "Allergy": "Immune system reaction to foreign substances like pollen or food.",
  "Arthritis": "Inflammation of joints causing pain, stiffness and swelling.",
  "Bronchial Asthma": "Chronic respiratory condition causing breathing difficulties.",
  "Cervical Spondylosis": "Age-related wear of the spinal discs in the neck.",
  "Chickenpox": "A highly contagious viral infection causing itchy blisters.",
  "Chronic Cholestasis": "Reduced bile flow from the liver causing itching and jaundice.",
  "Common Cold": "A viral infection of the upper respiratory tract.",
  "Dengue": "A mosquito-borne viral disease causing high fever and severe pain.",
  "Diabetes": "A metabolic disease causing high blood sugar levels.",
  "Dimorphic Hemmorhoids (piles)": "Swollen veins in the lower rectum or anus.",
  "Drug Reaction": "An adverse response of the body to a medication.",
  "Fungal Infection": "Infection caused by fungi, often affecting skin or nails.",
  "GERD": "Acid reflux disease where stomach acid flows back into the oesophagus.",
  "Gastroenteritis": "Inflammation of the stomach and intestines, causing vomiting and diarrhoea.",
  "Heart Attack": "Blockage of blood flow to the heart muscle.",
  "Hepatitis A": "A viral liver infection transmitted through contaminated food or water.",
  "Hepatitis B": "A viral liver infection transmitted through blood or bodily fluids.",
  "Hepatitis C": "A blood-borne viral infection that causes liver inflammation.",
  "Hepatitis D": "A liver infection that only occurs alongside Hepatitis B.",
  "Hepatitis E": "A waterborne viral liver infection, usually self-limiting.",
  "Hypertension": "A chronic condition of persistently elevated blood pressure.",
  "Hyperthyroidism": "Overactive thyroid gland producing excess thyroid hormone.",
  "Hypoglycemia": "Abnormally low blood sugar, often related to diabetes management.",
  "Hypothyroidism": "Underactive thyroid gland producing insufficient thyroid hormone.",
  "Impetigo": "A highly contagious bacterial skin infection causing sores.",
  "Jaundice": "Yellowing of skin and eyes caused by excess bilirubin in the blood.",
  "Malaria": "A mosquito-borne disease caused by Plasmodium parasites.",
  "Migraine": "A neurological condition causing intense, recurring headaches.",
  "Osteoarthritis": "Degenerative joint disease causing cartilage breakdown.",
  "Paralysis (brain hemorrhage)": "Loss of muscle function due to bleeding in or around the brain.",
  "Peptic Ulcer Disease": "Sores that develop on the stomach lining or small intestine.",
  "Pneumonia": "Infection that inflames air sacs in one or both lungs.",
  "Psoriasis": "A skin disease causing red, itchy scaly patches.",
  "Tuberculosis": "A serious bacterial infection primarily affecting the lungs.",
  "Typhoid": "A bacterial infection spread through contaminated food and water.",
  "Urinary Tract Infection": "Infection in any part of the urinary system.",
  "Varicose Veins": "Enlarged, twisted veins, usually in the legs.",
  "Vertigo": "A sensation of spinning or dizziness, often inner-ear related."
};

const DISEASE_PROFILES = JSON.parse(String.raw`{"AIDS":{"high_fever":0.9504,"extra_marital_contacts":0.9008,"muscle_wasting":0.9008,"patches_in_throat":0.9008},"Acne":{"skin_rash":0.9504,"blackheads":0.9008,"pus_filled_pimples":0.9008,"scurring":0.9008},"Alcoholic Hepatitis":{"abdominal_pain":0.9504,"distention_of_abdomen":0.9504,"fluid_overload":0.9504,"history_of_alcohol_consumption":0.9504,"swelling_of_stomach":0.9504,"vomiting":0.9504,"yellowish_skin":0.9504},"Allergy":{"chills":0.9008,"continuous_sneezing":0.9008,"shivering":0.9008,"watering_from_eyes":0.9008},"Arthritis":{"movement_stiffness":0.9504,"muscle_weakness":0.9504,"painful_walking":0.9504,"stiff_neck":0.9504,"swelling_joints":0.9504},"Bronchial Asthma":{"breathlessness":0.9504,"family_history":0.9504,"high_fever":0.9504,"mucoid_sputum":0.9504,"cough":0.9008,"fatigue":0.9008},"Cervical Spondylosis":{"dizziness":0.9504,"loss_of_balance":0.9504,"neck_pain":0.9504,"back_pain":0.9008,"weakness_in_limbs":0.9008},"Chickenpox":{"malaise":1.0,"red_spots_over_body":1.0,"fatigue":0.9504,"headache":0.9504,"high_fever":0.9504,"itching":0.9504,"lethargy":0.9504,"loss_of_appetite":0.9504,"mild_fever":0.9504,"skin_rash":0.9504,"swelled_lymph_nodes":0.9504},"Chronic Cholestasis":{"abdominal_pain":0.9504,"itching":0.9504,"loss_of_appetite":0.9504,"nausea":0.9504,"vomiting":0.9504,"yellowing_of_eyes":0.9504,"yellowish_skin":0.9504},"Common Cold":{"chest_pain":1.0,"congestion":1.0,"loss_of_smell":1.0,"muscle_pain":1.0,"phlegm":1.0,"redness_of_eyes":1.0,"runny_nose":1.0,"sinus_pressure":1.0,"throat_irritation":1.0,"chills":0.9504,"continuous_sneezing":0.9504,"cough":0.9504,"fatigue":0.9504,"headache":0.9504,"high_fever":0.9504,"malaise":0.9504,"swelled_lymph_nodes":0.9504},"Dengue":{"back_pain":1.0,"headache":1.0,"loss_of_appetite":1.0,"nausea":1.0,"pain_behind_the_eyes":1.0,"chills":0.9504,"fatigue":0.9504,"high_fever":0.9504,"joint_pain":0.9504,"malaise":0.9504,"muscle_pain":0.9504,"red_spots_over_body":0.9504,"skin_rash":0.9504,"vomiting":0.9504},"Diabetes":{"increased_appetite":1.0,"polyuria":1.0,"blurred_and_distorted_vision":0.9504,"excessive_hunger":0.9504,"fatigue":0.9504,"irregular_sugar_level":0.9504,"lethargy":0.9504,"obesity":0.9504,"restlessness":0.9504,"weight_loss":0.9504},"Dimorphic Hemmorhoids (piles)":{"bloody_stool":0.9504,"constipation":0.9504,"irritation_in_anus":0.9504,"pain_during_bowel_movements":0.9504,"pain_in_anal_region":0.9504},"Drug Reaction":{"itching":0.9504,"burning_micturition":0.9008,"skin_rash":0.9008,"spotting_ urination":0.9008,"stomach_pain":0.9008},"Fungal Infection":{"dischromic _patches":0.9008,"itching":0.9008,"nodal_skin_eruptions":0.9008,"skin_rash":0.9008},"GERD":{"chest_pain":0.9504,"cough":0.9504,"stomach_pain":0.9504,"acidity":0.9008,"ulcers_on_tongue":0.9008,"vomiting":0.9008},"Gastroenteritis":{"diarrhoea":0.9504,"dehydration":0.9008,"sunken_eyes":0.9008,"vomiting":0.9008},"Heart Attack":{"chest_pain":0.9504,"breathlessness":0.9008,"sweating":0.9008,"vomiting":0.9008},"Hepatitis A":{"mild_fever":1.0,"muscle_pain":1.0,"yellowing_of_eyes":1.0,"abdominal_pain":0.9504,"dark_urine":0.9504,"diarrhoea":0.9504,"joint_pain":0.9504,"loss_of_appetite":0.9504,"nausea":0.9504,"vomiting":0.9504,"yellowish_skin":0.9504},"Hepatitis B":{"malaise":1.0,"receiving_blood_transfusion":1.0,"receiving_unsterile_injections":1.0,"yellowing_of_eyes":1.0,"abdominal_pain":0.9504,"dark_urine":0.9504,"fatigue":0.9504,"itching":0.9504,"lethargy":0.9504,"loss_of_appetite":0.9504,"yellow_urine":0.9504,"yellowish_skin":0.9504},"Hepatitis C":{"family_history":0.9504,"fatigue":0.9504,"loss_of_appetite":0.9504,"nausea":0.9504,"yellowish_skin":0.9504,"yellowing_of_eyes":0.9008},"Hepatitis D":{"abdominal_pain":0.9504,"dark_urine":0.9504,"fatigue":0.9504,"joint_pain":0.9504,"loss_of_appetite":0.9504,"nausea":0.9504,"vomiting":0.9504,"yellowing_of_eyes":0.9504,"yellowish_skin":0.9504},"Hepatitis E":{"abdominal_pain":1.0,"coma":1.0,"loss_of_appetite":1.0,"stomach_bleeding":1.0,"yellowing_of_eyes":1.0,"acute_liver_failure":0.9504,"dark_urine":0.9504,"fatigue":0.9504,"high_fever":0.9504,"joint_pain":0.9504,"nausea":0.9504,"vomiting":0.9504,"yellowish_skin":0.9504},"Hypertension":{"lack_of_concentration":0.9504,"loss_of_balance":0.9504,"chest_pain":0.9008,"dizziness":0.9008,"headache":0.9008},"Hyperthyroidism":{"abnormal_menstruation":1.0,"irritability":1.0,"muscle_weakness":1.0,"diarrhoea":0.9504,"excessive_hunger":0.9504,"fast_heart_rate":0.9504,"fatigue":0.9504,"mood_swings":0.9504,"restlessness":0.9504,"sweating":0.9504,"weight_loss":0.9504},"Hypoglycemia":{"excessive_hunger":1.0,"irritability":1.0,"palpitations":1.0,"slurred_speech":1.0,"anxiety":0.9504,"blurred_and_distorted_vision":0.9504,"drying_and_tingling_lips":0.9504,"fatigue":0.9504,"headache":0.9504,"nausea":0.9504,"sweating":0.9504,"vomiting":0.9504},"Hypothyroidism":{"abnormal_menstruation":1.0,"brittle_nails":1.0,"depression":1.0,"enlarged_thyroid":1.0,"irritability":1.0,"swollen_extremeties":1.0,"cold_hands_and_feets":0.9504,"dizziness":0.9504,"lethargy":0.9504,"mood_swings":0.9504,"puffy_face_and_eyes":0.9504,"weight_gain":0.9504,"fatigue":0.9008},"Impetigo":{"blister":0.9504,"red_sore_around_nose":0.9504,"skin_rash":0.9504,"yellow_crust_ooze":0.9504,"high_fever":0.8512},"Jaundice":{"abdominal_pain":0.9504,"dark_urine":0.9504,"fatigue":0.9504,"high_fever":0.9504,"itching":0.9504,"vomiting":0.9504,"weight_loss":0.9504,"yellowish_skin":0.9504},"Malaria":{"muscle_pain":1.0,"chills":0.9504,"headache":0.9504,"high_fever":0.9504,"nausea":0.9504,"sweating":0.9504,"vomiting":0.9504,"diarrhoea":0.9008},"Migraine":{"acidity":0.9504,"blurred_and_distorted_vision":0.9504,"depression":0.9504,"excessive_hunger":0.9504,"headache":0.9504,"indigestion":0.9504,"irritability":0.9504,"stiff_neck":0.9504,"visual_disturbances":0.9504},"Osteoarthritis":{"hip_joint_pain":0.9504,"joint_pain":0.9504,"knee_pain":0.9504,"neck_pain":0.9504,"painful_walking":0.9504,"swelling_joints":0.9504},"Paralysis (brain hemorrhage)":{"altered_sensorium":0.9504,"headache":0.9008,"vomiting":0.9008,"weakness_of_one_body_side":0.9008},"Peptic Ulcer Disease":{"abdominal_pain":0.9504,"internal_itching":0.9504,"passage_of_gases":0.9504,"vomiting":0.9504,"indigestion":0.9008,"loss_of_appetite":0.9008},"Pneumonia":{"chest_pain":1.0,"fast_heart_rate":1.0,"rusty_sputum":1.0,"breathlessness":0.9504,"chills":0.9504,"cough":0.9504,"fatigue":0.9504,"high_fever":0.9504,"malaise":0.9504,"phlegm":0.9504,"sweating":0.9504},"Psoriasis":{"inflammatory_nails":0.9504,"joint_pain":0.9504,"silver_like_dusting":0.9504,"skin_peeling":0.9504,"skin_rash":0.9504,"small_dents_in_nails":0.9504},"Tuberculosis":{"blood_in_sputum":1.0,"chest_pain":1.0,"loss_of_appetite":1.0,"malaise":1.0,"mild_fever":1.0,"phlegm":1.0,"swelled_lymph_nodes":1.0,"yellowing_of_eyes":1.0,"breathlessness":0.9504,"chills":0.9504,"cough":0.9504,"fatigue":0.9504,"high_fever":0.9504,"sweating":0.9504,"vomiting":0.9504,"weight_loss":0.9504},"Typhoid":{"chills":1.0,"fatigue":1.0,"high_fever":1.0,"abdominal_pain":0.9504,"belly_pain":0.9504,"constipation":0.9504,"diarrhoea":0.9504,"headache":0.9504,"nausea":0.9504,"toxic_look_(typhos)":0.9504,"vomiting":0.9504},"Urinary Tract Infection":{"bladder_discomfort":0.9504,"continuous_feel_of_urine":0.9504,"burning_micturition":0.9008,"foul_smell_of urine":0.8512},"Varicose Veins":{"bruising":0.9504,"cramps":0.9504,"fatigue":0.9504,"obesity":0.9504,"prominent_veins_on_calf":0.9504,"swollen_legs":0.9504,"swollen_blood_vessels":0.9008},"Vertigo":{"headache":0.9504,"loss_of_balance":0.9504,"nausea":0.9504,"unsteadiness":0.9504,"vomiting":0.9504,"spinning_movements":0.9008}}`);

const PROFILE_TOTALS = Object.fromEntries(
  Object.entries(DISEASE_PROFILES).map(([disease, profile]) => [
    disease,
    Object.values(profile).reduce((sum, value) => sum + value, 0)
  ])
);
const API_BASE_URL = window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost"
  ? "http://127.0.0.1:8000"
  : "https://your-render-backend.onrender.com";
const API_URL = `${API_BASE_URL}/predict`;

const selected = new Set();
let currentCat = "all";
let currentSearch = "";

function toLabel(symptom) {
  return symptom
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .replace(/\./g, "")
    .trim()
    .replace(/\b\w/g, char => char.toUpperCase());
}

function toRawSymptom(label) {
  return SYMPTOMS.find(symptom => toLabel(symptom).toLowerCase() === label.toLowerCase());
}

function renderGrid() {
  const grid = document.getElementById("symptomGrid");
  const noResults = document.getElementById("noResults");
  const query = currentSearch.toLowerCase().trim();
  const pool = currentCat === "all" ? SYMPTOMS : (CATEGORIES[currentCat] || SYMPTOMS);
  const filtered = pool.filter(symptom => (
    !query ||
    toLabel(symptom).toLowerCase().includes(query) ||
    symptom.includes(query)
  ));

  if (filtered.length === 0) {
    grid.innerHTML = "";
    noResults.style.display = "block";
    return;
  }

  noResults.style.display = "none";
  grid.innerHTML = filtered.map(symptom => {
    const isSelected = selected.has(symptom);
    return `
      <div class="chip${isSelected ? " selected" : ""}" data-sym="${symptom}">
        <div class="chip-check">${isSelected ? "&#10003;" : ""}</div>
        <div class="chip-label">${toLabel(symptom)}</div>
      </div>
    `;
  }).join("");
}

function updateCounts() {
  const count = selected.size;
  document.getElementById("selBadge").textContent = `${count} selected`;
  document.getElementById("analyseBtn").disabled = count < 1;
}

function showState(state) {
  document.getElementById("emptyState").classList.toggle("hidden", state !== "empty");
  document.getElementById("loadingState").classList.toggle("active", state === "loading");
  document.getElementById("resultContent").classList.toggle("active", state === "result");
}

function scoreDisease(profile, selectedSymptoms) {
  const totalWeight = Object.values(profile).reduce((sum, value) => sum + value, 0);
  let matchedWeight = 0;
  let precisionPenalty = 0;
  let missingPenalty = 0;

  for (const symptom of selectedSymptoms) {
    const weight = profile[symptom];
    if (weight) {
      matchedWeight += weight;
      precisionPenalty += 1 - weight;
    } else {
      precisionPenalty += 1;
    }
  }

  for (const [symptom, weight] of Object.entries(profile)) {
    if (!selectedSymptoms.has(symptom)) missingPenalty += weight;
  }

  const coverage = totalWeight ? matchedWeight / totalWeight : 0;
  const precision = selectedSymptoms.size ? matchedWeight / selectedSymptoms.size : 0;
  const vectorSimilarity = 1 - ((precisionPenalty + missingPenalty) / SYMPTOMS.length);
  const confidence = Math.round(
    Math.max(0, Math.min(1, (coverage * 0.45) + (precision * 0.35) + (vectorSimilarity * 0.20))) * 100
  );

  return { confidence, coverage, precision, matchedWeight };
}

function buildLocalPrediction() {
  const ranked = Object.entries(DISEASE_PROFILES)
    .map(([disease, profile]) => ({
      disease,
      profile,
      ...scoreDisease(profile, selected)
    }))
    .sort((a, b) => (
      b.confidence - a.confidence ||
      b.matchedWeight - a.matchedWeight ||
      PROFILE_TOTALS[b.disease] - PROFILE_TOTALS[a.disease]
    ));

  const top = ranked[0];
  const top5 = ranked.slice(0, 5).map(item => ({
    disease: item.disease,
    score: item.confidence
  }));

  const suggestedSymptoms = Object.entries(top.profile)
    .filter(([symptom]) => !selected.has(symptom))
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([symptom]) => toLabel(symptom));

  return {
    top_disease: top.disease,
    confidence: top.confidence,
    top5,
    suggested_symptoms: suggestedSymptoms
  };
}

async function fetchPrediction() {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symptoms: [...selected] })
  });

  if (!response.ok) {
    throw new Error(`Prediction request failed with status ${response.status}`);
  }

  return response.json();
}

function renderResult(data) {
  const name = (data.top_disease || "Unknown").trim();
  const confidence = Math.min(100, Math.max(0, data.confidence || 0));

  document.getElementById("diagName").textContent = name;
  document.getElementById("diagDesc").textContent =
    DISEASE_DESC[name] || "A condition identified based on the selected symptoms.";
  document.getElementById("confPct").textContent = `${confidence}%`;
  document.getElementById("confFill").style.width = "0%";

  setTimeout(() => {
    document.getElementById("confFill").style.width = `${confidence}%`;
  }, 80);

  const rankStyles = ["gold", "silver", "bronze", "", ""];
  const top5 = data.top5 || [];
  const maxScore = top5[0]?.score || 1;

  document.getElementById("predList").innerHTML = top5.map((item, index) => {
    const barWidth = Math.round((item.score / maxScore) * 100);
    return `
      <div class="pred-item" style="animation-delay:${index * 55}ms">
        <div class="pred-rank ${rankStyles[index]}">#${index + 1}</div>
        <div class="pred-name">${item.disease}</div>
        <div class="pred-bar-track"><div class="pred-bar-fill" style="width:${barWidth}%"></div></div>
        <div class="pred-pct">${item.score}%</div>
      </div>
    `;
  }).join("");

  const suggestionRow = document.getElementById("sugRow");
  const suggestions = data.suggested_symptoms || [];

  suggestionRow.innerHTML = suggestions.map(label => {
    const raw = toRawSymptom(label);
    const used = raw ? selected.has(raw) : false;
    return `<span class="sug-chip${used ? " used" : ""}" data-raw="${raw || ""}">${label}</span>`;
  }).join("");

  suggestionRow.querySelectorAll(".sug-chip:not(.used)").forEach(chip => {
    chip.addEventListener("click", () => {
      const raw = chip.dataset.raw;
      if (!raw || selected.has(raw)) return;
      selected.add(raw);
      updateCounts();
      renderGrid();
      chip.classList.add("used");
    });
  });

  showState("result");
}

document.getElementById("symptomGrid").addEventListener("click", event => {
  const chip = event.target.closest(".chip");
  if (!chip) return;

  const symptom = chip.dataset.sym;
  if (selected.has(symptom)) selected.delete(symptom);
  else selected.add(symptom);

  updateCounts();
  renderGrid();
});

document.getElementById("catTabs").addEventListener("click", event => {
  const button = event.target.closest(".cat");
  if (!button) return;

  currentCat = button.dataset.cat;
  document.querySelectorAll(".cat").forEach(tab => tab.classList.remove("active"));
  button.classList.add("active");
  renderGrid();
});

document.getElementById("searchInput").addEventListener("input", event => {
  currentSearch = event.target.value;
  renderGrid();
});

document.getElementById("clearBtn").addEventListener("click", () => {
  selected.clear();
  updateCounts();
  renderGrid();
  showState("empty");
});

document.getElementById("analyseBtn").addEventListener("click", async () => {
  if (selected.size === 0) return;

  showState("loading");
  document.getElementById("analyseBtn").disabled = true;

  try {
    const prediction = await fetchPrediction();
    renderResult(prediction);
  } catch (error) {
    console.warn("Python API unavailable, using local fallback:", error);
    await new Promise(resolve => setTimeout(resolve, 280));
    renderResult(buildLocalPrediction());
  } finally {
    document.getElementById("analyseBtn").disabled = false;
  }
});

renderGrid();
updateCounts();
showState("empty");
