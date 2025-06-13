class ExcelMCPTool {
    constructor() {
        this.ws = null;
        this.chart = null;
        this.currentData = null;
        
        this.initializeEventListeners();
        this.connectWebSocket();
    }
    
    initializeEventListeners() {
        const uploadBtn = document.getElementById('uploadBtn');
        const fileInput = document.getElementById('fileInput');
        const sendBtn = document.getElementById('sendBtn');
        const commandInput = document.getElementById('commandInput');
        
        uploadBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        sendBtn.addEventListener('click', () => this.sendCommand());
        commandInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendCommand();
        });
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.addMessage('Connection error. Please refresh the page.', 'ai');
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            setTimeout(() => this.connectWebSocket(), 3000);
        };
    }
    
    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.currentData = result;
                this.displayFileInfo(file.name, result);
                this.displayDataPreview(result.preview);
                this.displayDataStats(result);
                this.addMessage(`File "${file.name}" uploaded successfully! You can now ask questions about your data.`, 'ai');
            } else {
                this.addMessage(`Error uploading file: ${result.error}`, 'ai');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.addMessage('Error uploading file. Please try again.', 'ai');
        }
    }
    
    displayFileInfo(filename, data) {
        const fileInfo = document.getElementById('fileInfo');
        fileInfo.innerHTML = `
            <div style="color: white;">
                <strong>📁 ${filename}</strong><br>
                Shape: ${data.shape[0]} rows × ${data.shape[1]} columns<br>
                Columns: ${data.columns.join(', ')}
            </div>
        `;
    }
    
    displayDataPreview(preview) {
        const dataTable = document.getElementById('dataTable');
        
        if (!preview || preview.length === 0) {
            dataTable.innerHTML = '<p style="color: white;">No data to preview</p>';
            return;
        }
        
        const columns = Object.keys(preview[0]);
        
        let tableHTML = '<table class="data-table"><thead><tr>';
        columns.forEach(col => {
            tableHTML += `<th>${col}</th>`;
        });
        tableHTML += '</tr></thead><tbody>';
        
        preview.forEach(row => {
            tableHTML += '<tr>';
            columns.forEach(col => {
                tableHTML += `<td>${row[col] || ''}</td>`;
            });
            tableHTML += '</tr>';
        });
        
        tableHTML += '</tbody></table>';
        dataTable.innerHTML = tableHTML;
    }
    
    displayDataStats(data) {
        const dataStats = document.getElementById('dataStats');
        dataStats.innerHTML = `
            <div style="color: white; font-size: 0.9rem;">
                <div style="margin-bottom: 10px;">
                    <strong>📊 Rows:</strong> ${data.shape[0]}<br>
                    <strong>📋 Columns:</strong> ${data.shape[1]}
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px;">
                    <strong>Column Types:</strong><br>
                    ${Object.entries(data.dtypes).map(([col, type]) => 
                        `${col}: ${type}`
                    ).join('<br>')}
                </div>
            </div>
        `;
    }
    
    sendCommand() {
        const input = document.getElementById('commandInput');
        const command = input.value.trim();
        
        if (!command) return;
        if (!this.currentData) {
            this.addMessage('Please upload a file first.', 'ai');
            return;
        }
        
        this.addMessage(command, 'user');
        input.value = '';
        
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ command }));
        } else {
            this.addMessage('Connection lost. Please refresh the page.', 'ai');
        }
    }
    
    handleWebSocketMessage(data) {
        if (data.status === 'processing') {
            this.addMessage(data.message, 'ai');
        } else if (data.status === 'complete') {
            this.handleAnalysisResult(data);
        } else if (data.error) {
            this.addMessage(`Error: ${data.error}`, 'ai');
        }
    }
    
    handleAnalysisResult(data) {
        const result = data.result;
        
        if (result.error) {
            this.addMessage(`Analysis error: ${result.error}`, 'ai');
            return;
        }
        
        // Display text result
        let message = 'Analysis complete!';
        
        if (result.count !== undefined) {
            message += ` Found ${result.count} matching records.`;
        }
        
        if (result.statistics) {
            message += '\n\nStatistics:\n' + JSON.stringify(result.statistics, null, 2);
        }
        
        this.addMessage(message, 'ai');
        
        // Create chart if data is suitable for visualization
        if (result.labels && result.values && data.instruction.chart_type) {
            this.createChart(result, data.instruction.chart_type);
        } else if (result.data && typeof result.data === 'object') {
            // Handle grouped data
            const labels = Object.keys(result.data);
            const values = Object.values(result.data);
            
            if (labels.length > 0 && data.instruction.chart_type) {
                this.createChart({ labels, values }, data.instruction.chart_type);
            }
        }
    }
    
    createChart(data, chartType) {
        const canvas = document.getElementById('chartCanvas');
        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart
        if (this.chart) {
            this.chart.destroy();
        }
        
        const config = {
            type: chartType,
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Data',
                    data: data.values,
                    backgroundColor: this.getChartColors(data.values.length),
                    borderColor: 'rgba(255, 255, 255, 0.8)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: 'white'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: 'white'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        ticks: {
                            color: 'white'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        };
        
        this.chart = new Chart(ctx, config);
    }
    
    getChartColors(count) {
        const colors = [
            'rgba(255, 99, 132, 0.8)',
            'rgba(54, 162, 235, 0.8)',
            'rgba(255, 205, 86, 0.8)',
            'rgba(75, 192, 192, 0.8)',
            'rgba(153, 102, 255, 0.8)',
            'rgba(255, 159, 64, 0.8)',
            'rgba(199, 199, 199, 0.8)',
            'rgba(83, 102, 255, 0.8)'
        ];
        
        const result = [];
        for (let i = 0; i < count; i++) {
            result.push(colors[i % colors.length]);
        }
        return result;
    }
    
    addMessage(message, sender) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        // Format message for better display
        if (message.includes('\n')) {
            messageDiv.innerHTML = message.replace(/\n/g, '<br>');
        } else {
            messageDiv.textContent = message;
        }
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new ExcelMCPTool();
});
