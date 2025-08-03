// Global variables
let uploadedData = null;
let analysisResults = null;
let currentCharts = {};

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const loadingSection = document.getElementById('loadingSection');
const analyticsDashboard = document.getElementById('analyticsDashboard');
const progressFill = document.getElementById('progressFill');

// File upload functionality
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    const allowedTypes = ['.csv', '.xlsx', '.xls'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
        showNotification('Please upload a CSV or Excel file', 'error');
        return;
    }
    
    fileName.textContent = file.name;
    fileInfo.style.display = 'block';
    uploadArea.style.display = 'none';
    
    // Store file for analysis
    uploadedData = file;
}

function removeFile() {
    uploadedData = null;
    fileInfo.style.display = 'none';
    uploadArea.style.display = 'block';
    fileInput.value = '';
}

async function analyzeData() {
    if (!uploadedData) {
        showNotification('Please upload a file first', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const formData = new FormData();
        formData.append('file', uploadedData);
        
        // Simulate progress
        simulateProgress();
        
        // Upload and analyze data
        const response = await fetch('/api/upload_and_analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Analysis failed');
        }
        
        analysisResults = await response.json();
        
        hideLoading();
        showAnalytics();
        generateCharts();
        populateSummaryCards();
        populateInventoryInsights();
        populateDataTable();
        
        showNotification('Analysis completed successfully!', 'success');
        
    } catch (error) {
        hideLoading();
        showNotification('Analysis failed: ' + error.message, 'error');
        console.error('Analysis error:', error);
    }
}

async function loadRetailData() {
    showLoading();
    
    try {
        // Simulate progress
        simulateProgress();
        
        // Load retail dataset analytics
        const response = await fetch('/api/load_retail_data', {
            method: 'GET'
        });
        
        if (!response.ok) {
            throw new Error('Failed to load retail data');
        }
        
        analysisResults = await response.json();
        
        hideLoading();
        showAnalytics();
        generateCharts();
        populateSummaryCards();
        populateInventoryInsights();
        populateDataTable();
        
        showNotification('Retail analytics loaded successfully!', 'success');
        
    } catch (error) {
        hideLoading();
        showNotification('Failed to load retail data: ' + error.message, 'error');
        console.error('Retail data error:', error);
    }
}

function simulateProgress() {
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) {
            progress = 90;
            clearInterval(interval);
        }
        progressFill.style.width = progress + '%';
    }, 200);
}

function showLoading() {
    loadingSection.style.display = 'block';
    analyticsDashboard.style.display = 'none';
}

function hideLoading() {
    loadingSection.style.display = 'none';
    progressFill.style.width = '100%';
}

function showAnalytics() {
    analyticsDashboard.style.display = 'block';
    populateForecastDropdowns();
}

// Chart generation functions
function generateCharts() {
    if (!analysisResults) return;
    
    generateSalesTrendChart();
    generateProductPerformanceChart();
    generateCategoryChart();
    generateForecastChart();
}

function generateSalesTrendChart() {
    const data = analysisResults.sales_trend || [];
    const period = document.getElementById('trendPeriod').value;
    
    let filteredData = data;
    if (period !== 'all' && data.length > 0) {
        const days = parseInt(period);
        filteredData = data.slice(-days);
    }
    
    if (filteredData.length === 0) {
        document.getElementById('salesTrendChart').innerHTML = '<p class="no-data">No sales trend data available</p>';
        return;
    }
    
    const trace = {
        x: filteredData.map(d => d.date),
        y: filteredData.map(d => d.sales),
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#667eea', width: 3 },
        marker: { size: 6, color: '#667eea' },
        fill: 'tonexty',
        fillcolor: 'rgba(102, 126, 234, 0.1)'
    };
    
    const layout = {
        title: 'Sales Trend Analysis',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Units Sold' },
        hovermode: 'closest',
        margin: { l: 50, r: 50, t: 50, b: 50 }
    };
    
    Plotly.newPlot('salesTrendChart', [trace], layout, {responsive: true});
}

function generateProductPerformanceChart() {
    const data = analysisResults.top_products || [];
    const count = parseInt(document.getElementById('topProductsCount').value);
    
    const topData = data.slice(0, count);
    
    if (topData.length === 0) {
        document.getElementById('productPerformanceChart').innerHTML = '<p class="no-data">No product performance data available</p>';
        return;
    }
    
    const trace = {
        x: topData.map(p => p.product_name || p.product_id),
        y: topData.map(p => p.units_sold || p.total_sales),
        type: 'bar',
        marker: {
            color: topData.map((_, i) => `hsl(${240 + i * 20}, 70%, 60%)`),
            line: { color: '#ffffff', width: 1 }
        }
    };
    
    const layout = {
        title: 'Top Performing Products',
        xaxis: { title: 'Products', tickangle: -45 },
        yaxis: { title: 'Units Sold' },
        margin: { l: 50, r: 50, t: 50, b: 100 }
    };
    
    Plotly.newPlot('productPerformanceChart', [trace], layout, {responsive: true});
}

function generateCategoryChart() {
    const data = analysisResults.category_performance || [];
    
    if (data.length === 0) {
        document.getElementById('categoryChart').innerHTML = '<p class="no-data">No category performance data available</p>';
        return;
    }
    
    const trace = {
        labels: data.map(c => c.category),
        values: data.map(c => c.sales),
        type: 'pie',
        hole: 0.4,
        marker: {
            colors: ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
        }
    };
    
    const layout = {
        title: 'Category Performance',
        margin: { l: 50, r: 50, t: 50, b: 50 }
    };
    
    Plotly.newPlot('categoryChart', [trace], layout, {responsive: true});
}

function generateForecastChart(selectedCategory, selectedProduct) {
    if (!selectedProduct) {
        document.getElementById('forecastChart').innerHTML = '<p class="no-data">Select a product to view forecast</p>';
        return;
    }
    
    // Get the product name for display
    const selectedProductElement = document.getElementById('forecastProduct');
    const selectedOption = selectedProductElement.options[selectedProductElement.selectedIndex];
    const productName = selectedOption ? selectedOption.text : selectedProduct;
    
    // For now, show a placeholder since forecast data needs to be generated
    document.getElementById('forecastChart').innerHTML = `
        <div style="text-align: center; padding: 2rem;">
            <h4>Forecast for ${productName}</h4>
            <p>Forecast data will be generated when you select a product.</p>
            <button onclick="generateForecastForProduct('${selectedProduct}')" class="btn btn-primary">
                Generate Forecast
            </button>
        </div>
    `;
}

async function generateForecastForProduct(productId) {
    const forecastCategory = document.getElementById('forecastCategory').value;
    const forecastProduct = document.getElementById('forecastProduct').value;
    // Try to get store_id if available (add logic if you have a store dropdown)
    let storeId = null;
    if (document.getElementById('forecastStore')) {
        storeId = document.getElementById('forecastStore').value;
    }
    const payload = {
        product_id: productId,
        forecast_days: 7
    };
    if (storeId) payload.store_id = storeId;
    try {
        const response = await fetch('/api/forecast', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            throw new Error('Forecast generation failed');
        }
        const forecastData = await response.json();
        // Get the product name for display
        const selectedProductElement = document.getElementById('forecastProduct');
        const selectedOption = selectedProductElement.options[selectedProductElement.selectedIndex];
        const productName = selectedOption ? selectedOption.text : productId;
        
        // Display the forecast as integers
        const forecastChart = document.getElementById('forecastChart');
        forecastChart.innerHTML = `
            <div style="text-align: center; padding: 1rem;">
                <h4>7-Day Forecast for ${productName}</h4>
                <div style="display: flex; justify-content: space-around; margin: 1rem 0;">
                    ${forecastData.predictions.map((pred, i) => `
                        <div style="text-align: center; padding: 0.5rem; background: #f0f0f0; border-radius: 8px; margin: 0 0.25rem;">
                            <div style="font-weight: bold; color: #667eea;">Day ${i+1}</div>
                            <div style="font-size: 1.2rem;">${Math.round(pred)}</div>
                        </div>
                    `).join('')}
                </div>
                <p style="color: #666; font-size: 0.9rem;">Predicted units to be sold</p>
            </div>
        `;
    } catch (error) {
        showNotification('Forecast generation failed: ' + error.message, 'error');
    }
}

// Summary cards population
function populateSummaryCards() {
    if (!analysisResults) return;
    
    const summary = analysisResults.summary || {};
    document.getElementById('totalProducts').textContent = summary.total_products ? summary.total_products.toLocaleString() : '0';
    document.getElementById('totalStores').textContent = summary.total_stores ? summary.total_stores.toLocaleString() : '0';
    document.getElementById('totalSales').textContent = summary.total_sales ? summary.total_sales.toLocaleString() : '0';
    document.getElementById('totalRevenue').textContent = summary.total_revenue ? '$' + summary.total_revenue.toLocaleString() : '$0';
}

// Inventory insights population
function populateInventoryInsights() {
    if (!analysisResults) return;
    
    const lowStockList = document.getElementById('lowStockList');
    const reorderList = document.getElementById('reorderList');
    const safetyStockList = document.getElementById('safetyStockList');
    
    // Low stock alerts - create sample data if not available
    const lowStock = analysisResults.inventory?.low_stock || [];
    lowStockList.innerHTML = lowStock.length > 0 ? 
        lowStock.map(item => `
            <div class="alert-item">
                <strong>${item.product_name}</strong><br>
                Current Stock: ${item.current_stock}<br>
                Required: ${item.required_stock}
            </div>
        `).join('') : 
        '<div class="alert-item">No low stock alerts at this time</div>';
    
    // Reorder recommendations
    const reorderRecs = analysisResults.inventory?.reorder_recommendations || [];
    reorderList.innerHTML = reorderRecs.length > 0 ?
        reorderRecs.map(item => `
            <div class="recommendation-item">
                <strong>${item.product_name}</strong><br>
                Reorder Quantity: ${item.reorder_quantity}<br>
                Expected Cost: $${item.expected_cost}
            </div>
        `).join('') :
        '<div class="recommendation-item">No reorder recommendations at this time</div>';
    
    // Safety stock analysis
    const safetyStock = analysisResults.inventory?.safety_stock || [];
    safetyStockList.innerHTML = safetyStock.length > 0 ?
        safetyStock.map(item => `
            <div class="analysis-item">
                <strong>${item.product_name}</strong><br>
                Safety Stock: ${item.safety_stock}<br>
                Reorder Point: ${item.reorder_point}
            </div>
        `).join('') :
        '<div class="analysis-item">No safety stock analysis available</div>';
}

// Data table population
function populateDataTable() {
    if (!analysisResults) return;
    
    const tableBody = document.getElementById('tableBody');
    
    // Populate table
    const products = analysisResults.products || [];
    tableBody.innerHTML = products.map(product => `
        <tr>
            <td>${product.product_id}</td>
            <td>${product.product_name}</td>
            <td>${product.total_sales ? product.total_sales.toLocaleString() : '0'}</td>
            <td>${product.avg_daily_sales ? Math.round(product.avg_daily_sales).toLocaleString() : '0'}</td>
            <td>${product.forecast_7_days ? Math.round(product.forecast_7_days).toLocaleString() : '0'}</td>
            <td>${product.current_stock || 'N/A'}</td>
            <td><span class="status-badge status-${(product.stock_status || 'N/A').toLowerCase()}">${product.stock_status || 'N/A'}</span></td>
        </tr>
    `).join('');
}

function populateForecastDropdowns() {
    const forecastCategory = document.getElementById('forecastCategory');
    const forecastProduct = document.getElementById('forecastProduct');
    if (!analysisResults || !analysisResults.products) return;
    
    // Debug: Log the products data to see what we're working with
    console.log('Products data for forecast dropdowns:', analysisResults.products);
    // Get unique categories, filter out empty and 'General' if real categories exist
    let categories = [...new Set(analysisResults.products.map(p => p.product_category || ''))];
    categories = categories.filter(cat => cat && cat.toLowerCase() !== 'general');
    if (categories.length === 0 && analysisResults.products.length > 0) {
        categories = ['General'];
    }
    forecastCategory.innerHTML = '<option value="">Select Category</option>' +
        categories.map(cat => `<option value="${cat}">${cat}</option>`).join('');
    function updateProductsForCategory() {
        const selectedCategory = forecastCategory.value;
        let filteredProducts = analysisResults.products;
        if (selectedCategory) {
            filteredProducts = filteredProducts.filter(p => p.product_category === selectedCategory);
        }
        const productOptions = filteredProducts.map(p => `<option value="${p.product_id}">${p.product_name}</option>`).join('');
        forecastProduct.innerHTML = '<option value="">Select Product</option>' + productOptions;
        
        // Debug: Log the filtered products
        console.log('Filtered products for dropdown:', filteredProducts);
        console.log('Product options HTML:', productOptions);
    }
    forecastCategory.onchange = updateProductsForCategory;
    updateProductsForCategory();
}

// Update forecast chart to use both category and product
function updateForecastChart() {
    const forecastCategory = document.getElementById('forecastCategory').value;
    const forecastProduct = document.getElementById('forecastProduct').value;
    generateForecastChart(forecastCategory, forecastProduct);
}

// Update forecast category (when category changes, update products)
function updateForecastCategory() {
    populateForecastDropdowns();
    updateForecastChart();
}

// Chart update functions
function updateTrendChart() {
    generateSalesTrendChart();
}

function updateProductChart() {
    generateProductPerformanceChart();
}

// Table filtering and sorting
function filterTable() {
    const input = document.getElementById('searchInput').value.toLowerCase();
    const table = document.getElementById('dataTable');
    const rows = table.getElementsByTagName('tr');
    
    for (let i = 1; i < rows.length; i++) {
        const row = rows[i];
        const cells = row.getElementsByTagName('td');
        let found = false;
        
        for (let j = 0; j < cells.length; j++) {
            if (cells[j].textContent.toLowerCase().includes(input)) {
                found = true;
                break;
            }
        }
        
        row.style.display = found ? '' : 'none';
    }
}

function sortTable() {
    const sortBy = document.getElementById('sortBy').value;
    const table = document.getElementById('dataTable');
    const tbody = table.getElementsByTagName('tbody')[0];
    const rows = Array.from(tbody.getElementsByTagName('tr'));
    
    rows.sort((a, b) => {
        const aValue = getCellValue(a, sortBy);
        const bValue = getCellValue(b, sortBy);
        
        if (sortBy === 'sales' || sortBy === 'forecast') {
            return parseFloat(bValue) - parseFloat(aValue);
        }
        return aValue.localeCompare(bValue);
    });
    
    rows.forEach(row => tbody.appendChild(row));
}

function getCellValue(row, sortBy) {
    const cells = row.getElementsByTagName('td');
    switch (sortBy) {
        case 'product': return cells[1].textContent;
        case 'sales': return cells[2].textContent.replace(/,/g, '');
        case 'forecast': return cells[4].textContent.replace(/,/g, '');
        default: return cells[0].textContent;
    }
}

// Export functions
function exportToCSV() {
    if (!analysisResults || !analysisResults.products) {
        showNotification('No data available for export', 'error');
        return;
    }
    
    const headers = ['Product ID', 'Product Name', 'Total Sales', 'Avg Daily Sales', '7-Day Forecast', 'Current Stock', 'Status'];
    const data = analysisResults.products.map(p => [
        p.product_id,
        p.product_name,
        p.total_sales || 0,
        Math.round(p.avg_daily_sales || 0),
        Math.round(p.forecast_7_days || 0),
        p.current_stock || 'N/A',
        p.stock_status || 'N/A'
    ]);
    
    const csvContent = [headers, ...data].map(row => row.join(',')).join('\n');
    downloadFile(csvContent, 'analytics_report.csv', 'text/csv');
    showNotification('CSV exported successfully!', 'success');
}

function exportToPDF() {
    showNotification('PDF export feature coming soon!', 'info');
}

function generateReport() {
    showNotification('Report generation feature coming soon!', 'info');
}

function downloadFile(content, filename, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
}

// Utility functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.classList.add('show');
    }, 100);
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Mobile menu toggle
const hamburger = document.querySelector('.hamburger');
const navMenu = document.querySelector('.nav-menu');

hamburger.addEventListener('click', () => {
    hamburger.classList.toggle('active');
    navMenu.classList.toggle('active');
});

// Close mobile menu when clicking on a link
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
        hamburger.classList.remove('active');
        navMenu.classList.remove('active');
    });
});

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    console.log('Analytics page loaded successfully!');
    
    // Add notification styles
    const style = document.createElement('style');
    style.textContent = `
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 1rem 2rem;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            z-index: 1000;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        }
        
        .notification.show {
            transform: translateX(0);
        }
        
        .notification-info {
            background: #3182ce;
        }
        
        .notification-success {
            background: #38a169;
        }
        
        .notification-error {
            background: #e53e3e;
        }
        
        .no-data {
            text-align: center;
            padding: 2rem;
            color: #666;
            font-style: italic;
        }
        
        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: #667eea;
            color: white;
        }
        
        .btn-primary:hover {
            background: #5a67d8;
        }
    `;
    document.head.appendChild(style);
}); 