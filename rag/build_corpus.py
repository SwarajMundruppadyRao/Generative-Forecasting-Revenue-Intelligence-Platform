"""
Build text corpus from training data for RAG
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json

from utils.config import BASE_DIR, CORPUS_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__, BASE_DIR / "logs" / "corpus.log")


class CorpusBuilder:
    """Build text corpus from sales data"""
    
    def __init__(self):
        self.documents = []
        self.metadata = []
    
    def create_store_summaries(self, data: pd.DataFrame) -> List[Dict]:
        """Create text summaries for each store"""
        summaries = []
        
        for store_id in data['Store'].unique():
            store_data = data[data['Store'] == store_id]
            
            # Basic statistics
            total_sales = store_data['Weekly_Sales'].sum()
            avg_sales = store_data['Weekly_Sales'].mean()
            std_sales = store_data['Weekly_Sales'].std()
            
            # Time-based analysis
            if 'Date' in store_data.columns:
                store_data['Date'] = pd.to_datetime(store_data['Date'])
                date_range = f"{store_data['Date'].min().strftime('%Y-%m-%d')} to {store_data['Date'].max().strftime('%Y-%m-%d')}"
            else:
                date_range = "Unknown"
            
            # Store type
            store_type = store_data['Type'].iloc[0] if 'Type' in store_data.columns else "Unknown"
            store_size = store_data['Size'].iloc[0] if 'Size' in store_data.columns else 0
            
            # Create summary text
            summary = f"""
Store {store_id} Summary:
- Type: {store_type}
- Size: {store_size:,} square feet
- Date Range: {date_range}
- Total Sales: ${total_sales:,.2f}
- Average Weekly Sales: ${avg_sales:,.2f}
- Sales Volatility (Std Dev): ${std_sales:,.2f}
- Number of Records: {len(store_data)}
"""
            
            # Department breakdown
            if 'Dept' in store_data.columns:
                dept_sales = store_data.groupby('Dept')['Weekly_Sales'].agg(['sum', 'mean', 'count'])
                top_depts = dept_sales.nlargest(5, 'sum')
                
                summary += "\nTop 5 Departments by Total Sales:\n"
                for dept_id, row in top_depts.iterrows():
                    summary += f"  - Department {dept_id}: ${row['sum']:,.2f} total, ${row['mean']:,.2f} average, {int(row['count'])} records\n"
            
            # Holiday impact
            if 'IsHoliday' in store_data.columns:
                holiday_sales = store_data[store_data['IsHoliday'] == True]['Weekly_Sales'].mean()
                regular_sales = store_data[store_data['IsHoliday'] == False]['Weekly_Sales'].mean()
                
                if not np.isnan(holiday_sales) and not np.isnan(regular_sales):
                    impact = ((holiday_sales - regular_sales) / regular_sales) * 100
                    summary += f"\nHoliday Impact: {impact:+.1f}% change in average sales\n"
            
            summaries.append({
                'text': summary.strip(),
                'metadata': {
                    'store_id': int(store_id),
                    'type': 'store_summary',
                    'total_sales': float(total_sales),
                    'avg_sales': float(avg_sales)
                }
            })
        
        logger.info(f"Created {len(summaries)} store summaries")
        return summaries
    
    def create_department_summaries(self, data: pd.DataFrame) -> List[Dict]:
        """Create text summaries for each department"""
        if 'Dept' not in data.columns:
            return []
        
        summaries = []
        
        for dept_id in data['Dept'].unique():
            dept_data = data[data['Dept'] == dept_id]
            
            # Statistics
            total_sales = dept_data['Weekly_Sales'].sum()
            avg_sales = dept_data['Weekly_Sales'].mean()
            num_stores = dept_data['Store'].nunique()
            
            # Create summary
            summary = f"""
Department {dept_id} Summary:
- Total Sales Across All Stores: ${total_sales:,.2f}
- Average Weekly Sales: ${avg_sales:,.2f}
- Number of Stores Selling: {num_stores}
- Total Records: {len(dept_data)}

Top Performing Stores for Department {dept_id}:
"""
            
            # Top stores for this department
            store_sales = dept_data.groupby('Store')['Weekly_Sales'].agg(['sum', 'mean'])
            top_stores = store_sales.nlargest(5, 'sum')
            
            for store_id, row in top_stores.iterrows():
                summary += f"  - Store {store_id}: ${row['sum']:,.2f} total, ${row['mean']:,.2f} average\n"
            
            summaries.append({
                'text': summary.strip(),
                'metadata': {
                    'dept_id': int(dept_id),
                    'type': 'department_summary',
                    'total_sales': float(total_sales),
                    'avg_sales': float(avg_sales)
                }
            })
        
        logger.info(f"Created {len(summaries)} department summaries")
        return summaries
    
    def create_trend_summaries(self, data: pd.DataFrame) -> List[Dict]:
        """Create summaries about sales trends"""
        if 'Date' not in data.columns:
            return []
        
        summaries = []
        data['Date'] = pd.to_datetime(data['Date'])
        data['YearMonth'] = data['Date'].dt.to_period('M')
        
        # Overall trend
        monthly_sales = data.groupby('YearMonth')['Weekly_Sales'].sum().reset_index()
        monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)
        
        summary = "Overall Sales Trends:\n\n"
        summary += "Monthly Total Sales:\n"
        
        for _, row in monthly_sales.tail(12).iterrows():
            summary += f"  - {row['YearMonth']}: ${row['Weekly_Sales']:,.2f}\n"
        
        # Identify trends
        if len(monthly_sales) > 1:
            recent_avg = monthly_sales.tail(3)['Weekly_Sales'].mean()
            older_avg = monthly_sales.head(3)['Weekly_Sales'].mean()
            
            if recent_avg > older_avg:
                trend = "increasing"
                change = ((recent_avg - older_avg) / older_avg) * 100
            else:
                trend = "decreasing"
                change = ((older_avg - recent_avg) / older_avg) * 100
            
            summary += f"\nOverall Trend: Sales are {trend} with approximately {change:.1f}% change\n"
        
        summaries.append({
            'text': summary.strip(),
            'metadata': {
                'type': 'trend_summary'
            }
        })
        
        logger.info("Created trend summaries")
        return summaries
    
    def build_corpus(self, data: pd.DataFrame) -> List[Dict]:
        """
        Build complete corpus from data
        
        Args:
            data: Training data
        
        Returns:
            List of documents with text and metadata
        """
        logger.info("Building corpus from data...")
        
        documents = []
        
        # Create different types of summaries
        documents.extend(self.create_store_summaries(data))
        documents.extend(self.create_department_summaries(data))
        documents.extend(self.create_trend_summaries(data))
        
        self.documents = documents
        
        logger.info(f"Built corpus with {len(documents)} documents")
        
        return documents
    
    def save_corpus(self, path: Optional[Path] = None):
        """Save corpus to file"""
        if path is None:
            path = CORPUS_DIR / 'corpus.json'
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.documents, f, indent=2)
        
        logger.info(f"Saved corpus to {path}")
    
    def load_corpus(self, path: Optional[Path] = None) -> List[Dict]:
        """Load corpus from file"""
        if path is None:
            path = CORPUS_DIR / 'corpus.json'
        
        with open(path, 'r') as f:
            self.documents = json.load(f)
        
        logger.info(f"Loaded corpus from {path}")
        return self.documents


def main():
    """Build and save corpus"""
    # Load data
    data = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'train_processed.csv')
    
    # Build corpus
    builder = CorpusBuilder()
    documents = builder.build_corpus(data)
    
    # Save corpus
    builder.save_corpus()
    
    print(f"\nBuilt corpus with {len(documents)} documents")
    print("\nSample document:")
    print(documents[0]['text'][:500])


if __name__ == "__main__":
    main()
