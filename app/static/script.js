// DOM 요소 선택
const questionInput = document.getElementById('questionInput');
const askButton = document.getElementById('askButton');
const modelTypeOptions = document.querySelectorAll('input[name="modelType"]');
const loadingSpinner = document.getElementById('loadingSpinner');
const singleResponseCard = document.getElementById('singleResponseCard');
const singleResponseTitle = document.getElementById('singleResponseTitle');
const singleResponseContent = document.getElementById('singleResponseContent');
const comparisonCard = document.getElementById('comparisonCard');
const langchainResponse = document.getElementById('langchainResponse');
const graphragResponse = document.getElementById('graphragResponse');
const runEvaluationBtn = document.getElementById('runEvaluationBtn');
const sampleCount = document.getElementById('sampleCount');
const evaluationResult = document.getElementById('evaluationResult');
const evaluationDetails = document.getElementById('evaluationDetails');

// API 기본 URL
const API_BASE_URL = 'http://localhost:8000';

// 이벤트 리스너 설정
document.addEventListener('DOMContentLoaded', () => {
    askButton.addEventListener('click', handleAskButtonClick);
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleAskButtonClick();
        }
    });
    runEvaluationBtn.addEventListener('click', runEvaluation);
});

// 질문 처리 함수
async function handleAskButtonClick() {
    const question = questionInput.value.trim();
    if (!question) {
        alert('질문을 입력해주세요.');
        return;
    }

    // 선택된 모델 타입
    let selectedOption;
    modelTypeOptions.forEach(option => {
        if (option.checked) {
            selectedOption = option.value;
        }
    });

    // UI 상태 업데이트
    showLoading(true);
    hideResponseCards();

    try {
        if (selectedOption === 'compare') {
            // 비교 모드
            await handleCompareMode(question);
        } else {
            // 단일 모델 모드
            await handleSingleMode(question, selectedOption);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('오류가 발생했습니다: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 단일 모델 모드 처리
async function handleSingleMode(question, modelType) {
    const response = await fetch(`${API_BASE_URL}/question`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question: question,
            mode: modelType
        }),
    });

    const data = await response.json();
    
    if (response.ok) {
        // 응답 표시
        singleResponseTitle.textContent = modelType === 'langchain' ? 'LangChain 응답' : 'GraphRAG 응답';
        singleResponseContent.textContent = data.answer;
        singleResponseCard.classList.remove('d-none');
    } else {
        alert('오류가 발생했습니다: ' + data.detail);
    }
}

// 비교 모드 처리
async function handleCompareMode(question) {
    const response = await fetch(`${API_BASE_URL}/compare`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question: question
        }),
    });

    const data = await response.json();
    
    if (response.ok) {
        // 응답 표시
        langchainResponse.textContent = data.langchain_answer;
        graphragResponse.textContent = data.graphrag_answer;
        comparisonCard.classList.remove('d-none');
    } else {
        alert('오류가 발생했습니다: ' + data.detail);
    }
}

// 평가 실행 함수
async function runEvaluation() {
    const count = parseInt(sampleCount.value);
    if (isNaN(count) || count < 5 || count > 50) {
        alert('평가 질문 수는 5~50 사이로 입력해주세요.');
        return;
    }

    showLoading(true);
    evaluationResult.classList.add('d-none');

    try {
        const response = await fetch(`${API_BASE_URL}/evaluate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sample_count: count
            }),
        });

        const data = await response.json();
        
        if (response.ok) {
            // 평가 결과 표시
            displayEvaluationResults(data.results);
        } else {
            alert('오류가 발생했습니다: ' + data.detail);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('오류가 발생했습니다: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 평가 결과 표시 함수
function displayEvaluationResults(results) {
    evaluationResult.classList.remove('d-none');
    
    // 차트 데이터 준비
    const labels = [];
    const langchainData = [];
    const graphragData = [];
    
    for (const metric in results.LangChain) {
        labels.push(metric);
        langchainData.push(results.LangChain[metric]);
        graphragData.push(results.GraphRAG[metric]);
    }
    
    // 차트 그리기
    const ctx = document.getElementById('evaluationChart').getContext('2d');
    if (window.evaluationChartInstance) {
        window.evaluationChartInstance.destroy();
    }
    
    window.evaluationChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'LangChain',
                    data: langchainData,
                    backgroundColor: 'rgba(13, 110, 253, 0.7)',
                    borderColor: 'rgba(13, 110, 253, 1)',
                    borderWidth: 1
                },
                {
                    label: 'GraphRAG',
                    data: graphragData,
                    backgroundColor: 'rgba(25, 135, 84, 0.7)',
                    borderColor: 'rgba(25, 135, 84, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: '성능 비교 결과'
                }
            }
        }
    });
    
    // 상세 결과 표시
    let detailsHtml = '<table class="table table-striped">';
    detailsHtml += `
        <thead>
            <tr>
                <th>메트릭</th>
                <th>LangChain</th>
                <th>GraphRAG</th>
                <th>차이</th>
            </tr>
        </thead>
        <tbody>
    `;
    
    for (const metric in results.LangChain) {
        const langchainScore = results.LangChain[metric];
        const graphragScore = results.GraphRAG[metric];
        const diff = results.diff[metric];
        const diffClass = diff > 0 ? 'text-success' : (diff < 0 ? 'text-danger' : 'text-muted');
        const diffIcon = diff > 0 ? '↑' : (diff < 0 ? '↓' : '');
        
        detailsHtml += `
            <tr>
                <td>${metric}</td>
                <td>${langchainScore.toFixed(3)}</td>
                <td>${graphragScore.toFixed(3)}</td>
                <td class="${diffClass}">${diff.toFixed(3)} ${diffIcon}</td>
            </tr>
        `;
    }
    
    detailsHtml += '</tbody></table>';
    evaluationDetails.innerHTML = detailsHtml;
}

// UI 유틸리티 함수
function showLoading(show) {
    if (show) {
        loadingSpinner.classList.remove('d-none');
    } else {
        loadingSpinner.classList.add('d-none');
    }
}

function hideResponseCards() {
    singleResponseCard.classList.add('d-none');
    comparisonCard.classList.add('d-none');
}
