function searchPatents() {
    const params = {
        author: document.getElementById('author').value.trim(),
        applicant: document.getElementById('applicant').value.trim(),
        title: document.getElementById('title').value.trim(),
        province: document.getElementById('province').value.trim(),
        pub_no: document.getElementById('pub_no').value.trim(),
        pub_date: document.getElementById('pub_date').value.trim(),
        classification: document.getElementById('CLC').value.trim() // Matching param name
    };

    const query = new URLSearchParams(params).toString();
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '<p>正在加载推荐结果...</p>'; // Message

    fetch(`http://localhost:5000/api/recommend?${query}`)
        .then(response => response.json())
        .then(data => {
            // Logic
            if (data.code === 200 && data.data && data.data.length > 0) {
                renderResults(data.data);
            } else {
                container.innerHTML = `<p>${data.message || '未找到相关专利。'}</p>`;
            }
        })
        .catch(error => {
            console.error('请求失败:', error);
            // Error message
            container.innerHTML = '<p style="color:red;">请求失败，请检查后端是否启动。</p>';
        });
}

function loadClusterSummary() {
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '<p>正在加载聚类总览及可视化结果...</p>'; // 更新加载提示信息

    // 图片在根目录下的 model/ 文件夹中
    const imageUrl = "../models/clustering_visualization.png"; 

    fetch(`http://localhost:5000/api/clusters`)
        .then(res => res.json())
        .then(data => {
            let htmlContent = ''; // 用于构建最终的 HTML

            // 首先添加可视化图片的容器和标题
            // 即使 API 调用失败，我们也可以尝试显示图片（如果需要），或者只在成功时显示
            // 这里我们选择在 API 成功后一起构建显示内容

            if (data.code === 200 && data.data) {
                const summary = data.data;
                
                htmlContent += `<h2>聚类总览与可视化</h2>`;
                
                // 添加图片展示区域
                htmlContent += `<div class="visualization-container" style="margin-bottom: 20px; padding:10px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;">`;
                htmlContent += `  <h4>聚类可视化结果</h4>`; // 给图片一个小标题
                htmlContent += `  <img src="${imageUrl}" alt="聚类可视化图" style="max-width: 60%; height: auto; border: 1px solid #eee; border-radius: 4px; margin-top:10px;">`;
                htmlContent += `</div>`;

                // 添加聚类总览文本信息
                htmlContent += `<div class="cluster-summary" style="margin-top: 20px;"><h3>各聚类专利数量</h3>`; // 更新了H3标题
                if (Object.keys(summary).length === 0) {
                    htmlContent += '<p>暂无聚类信息。</p>';
                } else {
                    for (const [cid, count] of Object.entries(summary)) {
                        htmlContent += `<p><strong>聚类 ${cid}：</strong>包含 ${count} 条专利</p>`; // 稍作美化
                    }
                }
                htmlContent += `</div>`;
                
                container.innerHTML = htmlContent;

            } else {
                // API 数据获取失败，但仍然可以尝试显示图片，或者显示错误信息
                // 如果希望即使API失败也尝试显示图片，可以将图片HTML的构建移到if/else之外
                // 这里我们遵循原逻辑，API失败则显示错误
                container.innerHTML = `<h2>聚类总览与可视化</h2><p>${data.message || '加载聚类总览信息失败。'}</p><p>请同时检查可视化图片路径是否正确。</p>`;
            }
        })
        .catch(err => {
            console.error('加载聚类总览失败:', err);
            // 网络错误或其他错误，提示检查后端及图片
            container.innerHTML = `<h2>聚类总览与可视化</h2><p style="color:red;">加载失败，请检查后端接口或可视化图片（路径：${imageUrl}）是否存在。</p>`;
        });
}

function viewCluster() {
    const cid = document.getElementById('cluster_id').value.trim();
    const container = document.getElementById('resultsContainer');

    if (!cid) {
        // Message
        container.innerHTML = '<p style="color:red;">请输入聚类编号。</p>';
        return;
    }

    // Message
    container.innerHTML = `<p>正在加载聚类 ${cid} 的专利...</p>`;

    fetch(`http://localhost:5000/api/cluster?cluster_id=${cid}`) // Endpoint
        .then(res => res.json())
        .then(data => {
            // Logic
            if (data.code === 200 && data.data) {
                renderClusterResults(data.data, cid); // Pass cid for context if needed in title
            } else {
                container.innerHTML = `<p>${data.message || '加载聚类专利失败。'}</p>`;
            }
        })
        .catch(err => {
            console.error(`加载聚类 ${cid} 失败:`, err);
            // Error message
            container.innerHTML = '<p style="color:red;">加载失败，请检查后端接口。</p>';
        });
}

function renderResults(patents) {
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '<h2>搜索结果</h2>'; // Add a title for the results section

    if (patents.length === 0) {
        container.innerHTML += '<p>未找到符合条件的专利。</p>';
        return;
    }

    patents.forEach(patent => {
        const card = document.createElement('div');
        card.className = 'patent-card';
        // Content structure
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
        // Content structure
        card.innerHTML = `
            <h3>${patent.title || '无标题'}</h3>
            <p><strong>公开号：</strong>${patent.pub_no || '无'}</p>
            <p><strong>摘要：</strong>${patent.abstract || '无摘要'}</p>
        `;
        container.appendChild(card);
    });
}