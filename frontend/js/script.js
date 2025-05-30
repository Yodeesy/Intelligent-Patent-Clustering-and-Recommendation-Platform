document.addEventListener('DOMContentLoaded', function() {
    const notification = document.getElementById('notification-container');
    const searchButton = document.getElementById('searchButton');
    const viewClusterButton = document.getElementById('viewClusterButton');
    const loadClusterSummaryButton = document.getElementById('loadClusterSummaryButton');

    function showNotification(message) {
        console.log('showNotification 被调用，消息是:', message); // 添加这行

        if (notification) { // 确保元素存在
            notification.textContent = message;
            notification.classList.add('show');
            setTimeout(() => {
                notification.classList.remove('show');
            }, 1500);
        } else {
            console.error('找不到 notification-container 元素！');
        }
    }

    function searchPatents() {
        const params = {
            author: document.getElementById('author').value.trim(),
            applicant: document.getElementById('applicant').value.trim(),
            title: document.getElementById('title').value.trim(),
            province: document.getElementById('province').value.trim(),
            pub_no: document.getElementById('pub_no').value.trim(),
            pub_date: document.getElementById('pub_date').value.trim(),
            classification: document.getElementById('CLC').value.trim()
        };

        const query = new URLSearchParams(params).toString();
        const container = document.getElementById('resultsContainer');
        container.innerHTML = '<p>正在加载推荐结果...</p>';

        fetch(`http://localhost:5000/api/recommend?${query}`)
            .then(response => response.json())
            .then(data => {
                if (data.code === 200 && data.data && data.data.length > 0) {
                    renderResults(data.data);
                    showNotification('搜索成功！');
                } else {
                    container.innerHTML = `<p>${data.message || '未找到相关专利。'}</p>`;
                }
            })
            .catch(error => {
                console.error('请求失败:', error);
                container.innerHTML = '<p style="color:red;">请求失败，请检查后端是否启动。</p>';
            });
    }

    function loadClusterSummary() {
        const container = document.getElementById('resultsContainer');
        container.innerHTML = '<p>正在加载聚类总览及可视化结果...</p>';

        const imageUrl = "../models/clustering_visualization.png";

        fetch(`http://localhost:5000/api/clusters`)
            .then(res => res.json())
            .then(data => {
                let htmlContent = '';

                if (data.code === 200 && data.data) {
                    const summary = data.data;

                    htmlContent += `<h2>聚类总览与可视化</h2>`;
                    htmlContent += `<div class="visualization-container" style="margin-bottom: 20px; padding:10px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;">`;
                    htmlContent += `  <h4>聚类可视化结果</h4>`;
                    htmlContent += `  <img src="${imageUrl}" alt="聚类可视化图" style="max-width: 60%; height: auto; border: 1px solid #eee; border-radius: 4px; margin-top:10px;">`;
                    htmlContent += `</div>`;

                    htmlContent += `<div class="cluster-summary" style="margin-top: 20px;"><h3>各聚类专利数量</h3>`;
                    if (Object.keys(summary).length === 0) {
                        htmlContent += '<p>暂无聚类信息。</p>';
                    } else {
                        for (const [cid, count] of Object.entries(summary)) {
                            htmlContent += `<p><strong>聚类 ${cid}：</strong>包含 ${count} 条专利</p>`;
                        }
                    }
                    htmlContent += `</div>`;

                    container.innerHTML = htmlContent;
                    showNotification('加载总览成功！');
                } else {
                    container.innerHTML = `<h2>聚类总览与可视化</h2><p>${data.message || '加载聚类总览信息失败。'}</p><p>请同时检查可视化图片路径是否正确。</p>`;
                }
            })
            .catch(err => {
                console.error('加载聚类总览失败:', err);
                container.innerHTML = `<h2>聚类总览与可视化</h2><p style="color:red;">加载失败，请检查后端接口或可视化图片（路径：${imageUrl}）是否存在。</p>`;
            });
    }

    function viewCluster() {
        const cid = document.getElementById('cluster_id').value.trim();
        const container = document.getElementById('resultsContainer');

        if (!cid) {
            container.innerHTML = '<p style="color:red;">请输入聚类编号。</p>';
            return;
        }

        container.innerHTML = `<p>正在加载聚类 ${cid} 的专利...</p>`;

        fetch(`http://localhost:5000/api/cluster?cluster_id=${cid}`)
            .then(res => res.json())
            .then(data => {
                if (data.code === 200 && data.data) {
                    renderClusterResults(data.data, cid);
                    showNotification(`加载聚类 ${cid} 成功！`);
                } else {
                    container.innerHTML = `<p>${data.message || '加载聚类专利失败。'}</p>`;
                }
            })
            .catch(err => {
                console.error(`加载聚类 ${cid} 失败:`, err);
                container.innerHTML = '<p style="color:red;">加载失败，请检查后端接口。</p>';
            });
    }

    function renderResults(patents) {
        const container = document.getElementById('resultsContainer');
        container.innerHTML = '<h2>搜索结果</h2>';

        if (patents.length === 0) {
            container.innerHTML += '<p>未找到符合条件的专利。</p>';
            return;
        }

        patents.forEach(patent => {
            const card = document.createElement('div');
            card.className = 'patent-card';
            card.innerHTML = `
                <h3>${patent.title || '无标题'}</h3>
                <p><strong>数据来源：</strong>${patent.SrcDatabase || '未知'}</p>
                <p><strong>中图分类号：</strong>${patent.CLC || '无'}</p>
                <p><strong>作者：</strong>${patent.author || '无'}</p>
                <p><strong>申请人：</strong>${patent.applicant || '无'}</p>
                <p><strong>国省名称：</strong>${patent.province || '无'}</p>
                <p><strong>公开号：</strong>${patent.pub_no || '无'}</p>
                <p><strong>公开日期：</strong>${patent.pub_date || '无'}</p>
                <p><strong>类别：</strong>${patent.classification || '无'}</p>
                <p><strong>摘要：</strong>${patent.abstract || '无摘要'}</p>
                <p><strong>主权项：</strong>${patent.claims || '无主权项'}</p>
                <div class="similarity">相似度：${patent.similarity !== undefined ? (patent.similarity * 100).toFixed(2) + '%' : 'N/A'}</div>
            `;
            container.appendChild(card);
        });
    }

    function renderClusterResults(patents, clusterId) {
        const container = document.getElementById('resultsContainer');
        container.innerHTML = `<h2>聚类 ${clusterId} 的专利列表</h2>`;

        if (patents.length === 0) {
            container.innerHTML += `<p>该聚类下暂无专利。</p>`;
            return;
        }

        patents.forEach(patent => {
            const card = document.createElement('div');
            card.className = 'patent-card';
            card.innerHTML = `
                <h3>${patent.title || '无标题'}</h3>
                <p><strong>公开号：</strong>${patent.pub_no || '无'}</p>
                <p><strong>摘要：</strong>${patent.abstract || '无摘要'}</p>
            `;
            container.appendChild(card);
        });
    }

    // 添加事件监听器
    if (searchButton) {
        searchButton.addEventListener('click', searchPatents);
    }
    if (loadClusterSummaryButton) {
        loadClusterSummaryButton.addEventListener('click', loadClusterSummary);
    }
    if (viewClusterButton) {
        viewClusterButton.addEventListener('click', viewCluster);
    }

});