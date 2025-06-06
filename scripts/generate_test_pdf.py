from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import os

def create_test_pdf(output_path):
    """创建一个包含测试内容的PDF文档"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建PDF文档
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # 获取样式
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    
    # 创建内容
    story = []
    
    # 标题
    story.append(Paragraph("SunDB Database Manual", title_style))
    story.append(Spacer(1, 12))
    
    # 简介
    story.append(Paragraph("Introduction", styles['Heading2']))
    story.append(Paragraph(
        "SunDB is a high-performance distributed database system designed for modern applications. "
        "This manual provides comprehensive documentation on SunDB's features, configuration, and best practices.",
        styles['Normal']
    ))
    story.append(Spacer(1, 12))
    
    # 复制功能
    story.append(Paragraph("Replication", styles['Heading2']))
    story.append(Paragraph(
        "SunDB provides robust replication capabilities to ensure data availability and reliability. "
        "The replication system uses a master-slave architecture with automatic failover support.",
        styles['Normal']
    ))
    story.append(Spacer(1, 12))
    
    # 复制配置
    story.append(Paragraph("Replication Configuration", styles['Heading3']))
    replication_config = [
        ['Parameter', 'Default Value', 'Description'],
        ['replication.threads', '4', 'Number of replication threads'],
        ['replication.batch_size', '1000', 'Records per batch'],
        ['replication.timeout', '30s', 'Replication timeout'],
        ['replication.retry_interval', '5s', 'Retry interval on failure']
    ]
    
    t = Table(replication_config, colWidths=[2*inch, 1.5*inch, 3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 12))
    
    # 性能优化
    story.append(Paragraph("Performance Optimization", styles['Heading2']))
    story.append(Paragraph(
        "SunDB includes several features for performance optimization:",
        styles['Normal']
    ))
    story.append(Paragraph(
        "• Query optimization with cost-based optimizer<br/>"
        "• Automatic index selection<br/>"
        "• Memory management and caching<br/>"
        "• Parallel query execution",
        styles['Normal']
    ))
    story.append(Spacer(1, 12))
    
    # 安全特性
    story.append(Paragraph("Security Features", styles['Heading2']))
    story.append(Paragraph(
        "SunDB implements comprehensive security measures including:",
        styles['Normal']
    ))
    story.append(Paragraph(
        "• Role-based access control<br/>"
        "• Data encryption at rest and in transit<br/>"
        "• Audit logging<br/>"
        "• SSL/TLS support",
        styles['Normal']
    ))
    
    # 构建PDF
    doc.build(story)

if __name__ == '__main__':
    # 创建测试文档
    output_path = "./docs/SunDB_manual.pdf"
    create_test_pdf(output_path)
    print(f"Test PDF created at: {output_path}") 