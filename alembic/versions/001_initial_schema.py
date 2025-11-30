"""Initial database schema

Revision ID: 001
Revises: 
Create Date: 2024-11-30 12:00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create stocks table
    op.create_table('stocks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('exchange', sa.String(length=50), nullable=True),
        sa.Column('sector', sa.String(length=100), nullable=True),
        sa.Column('industry', sa.String(length=100), nullable=True),
        sa.Column('market_cap', sa.Float(), nullable=True),
        sa.Column('outstanding_shares', sa.Float(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_stocks_id'), 'stocks', ['id'], unique=False)
    op.create_index(op.f('ix_stocks_symbol'), 'stocks', ['symbol'], unique=True)
    
    # Create stock_prices table
    op.create_table('stock_prices',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stock_id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('open', sa.Float(), nullable=False),
        sa.Column('high', sa.Float(), nullable=False),
        sa.Column('low', sa.Float(), nullable=False),
        sa.Column('close', sa.Float(), nullable=False),
        sa.Column('volume', sa.Float(), nullable=False),
        sa.Column('adjusted_close', sa.Float(), nullable=True),
        sa.Column('change_percent', sa.Float(), nullable=True),
        sa.Column('source', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['stock_id'], ['stocks.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('stock_id', 'date', name='uq_stock_date')
    )
    op.create_index(op.f('ix_stock_prices_id'), 'stock_prices', ['id'], unique=False)
    op.create_index(op.f('ix_stock_prices_date'), 'stock_prices', ['date'], unique=False)
    op.create_index('ix_stock_prices_stock_date', 'stock_prices', ['stock_id', 'date'], unique=False)
    
    # Create technical_indicators table
    op.create_table('technical_indicators',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stock_id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('sma_20', sa.Float(), nullable=True),
        sa.Column('sma_50', sa.Float(), nullable=True),
        sa.Column('sma_200', sa.Float(), nullable=True),
        sa.Column('ema_12', sa.Float(), nullable=True),
        sa.Column('ema_26', sa.Float(), nullable=True),
        sa.Column('rsi_14', sa.Float(), nullable=True),
        sa.Column('macd', sa.Float(), nullable=True),
        sa.Column('macd_signal', sa.Float(), nullable=True),
        sa.Column('macd_histogram', sa.Float(), nullable=True),
        sa.Column('bb_upper', sa.Float(), nullable=True),
        sa.Column('bb_middle', sa.Float(), nullable=True),
        sa.Column('bb_lower', sa.Float(), nullable=True),
        sa.Column('atr_14', sa.Float(), nullable=True),
        sa.Column('obv', sa.Float(), nullable=True),
        sa.Column('adx_14', sa.Float(), nullable=True),
        sa.Column('calculated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['stock_id'], ['stocks.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('stock_id', 'date', name='uq_indicator_stock_date')
    )
    op.create_index(op.f('ix_technical_indicators_id'), 'technical_indicators', ['id'], unique=False)
    op.create_index(op.f('ix_technical_indicators_date'), 'technical_indicators', ['date'], unique=False)
    op.create_index('ix_indicators_stock_date', 'technical_indicators', ['stock_id', 'date'], unique=False)
    
    # Create sentiment_analysis table
    op.create_table('sentiment_analysis',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stock_id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('sentiment_score', sa.Float(), nullable=True),
        sa.Column('sentiment_label', sa.String(length=20), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('news_count', sa.Integer(), nullable=True),
        sa.Column('positive_count', sa.Integer(), nullable=True),
        sa.Column('negative_count', sa.Integer(), nullable=True),
        sa.Column('neutral_count', sa.Integer(), nullable=True),
        sa.Column('sources', sa.JSON(), nullable=True),
        sa.Column('top_headlines', sa.JSON(), nullable=True),
        sa.Column('model_name', sa.String(length=100), nullable=True),
        sa.Column('analyzed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['stock_id'], ['stocks.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('stock_id', 'date', name='uq_sentiment_stock_date')
    )
    op.create_index(op.f('ix_sentiment_analysis_id'), 'sentiment_analysis', ['id'], unique=False)
    op.create_index(op.f('ix_sentiment_analysis_date'), 'sentiment_analysis', ['date'], unique=False)
    op.create_index('ix_sentiment_stock_date', 'sentiment_analysis', ['stock_id', 'date'], unique=False)
    
    # Create model_metrics table
    op.create_table('model_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('stock_symbol', sa.String(length=20), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=True),
        sa.Column('mae', sa.Float(), nullable=True),
        sa.Column('rmse', sa.Float(), nullable=True),
        sa.Column('mape', sa.Float(), nullable=True),
        sa.Column('r2_score', sa.Float(), nullable=True),
        sa.Column('train_start_date', sa.Date(), nullable=True),
        sa.Column('train_end_date', sa.Date(), nullable=True),
        sa.Column('test_start_date', sa.Date(), nullable=True),
        sa.Column('test_end_date', sa.Date(), nullable=True),
        sa.Column('training_samples', sa.Integer(), nullable=True),
        sa.Column('test_samples', sa.Integer(), nullable=True),
        sa.Column('hyperparameters', sa.JSON(), nullable=True),
        sa.Column('model_file_path', sa.String(length=500), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('trained_at', sa.DateTime(), nullable=True),
        sa.Column('training_duration_seconds', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_model_metrics_id'), 'model_metrics', ['id'], unique=False)
    op.create_index(op.f('ix_model_metrics_stock_symbol'), 'model_metrics', ['stock_symbol'], unique=False)
    op.create_index('ix_model_active', 'model_metrics', ['model_name', 'stock_symbol', 'is_active'], unique=False)
    
    # Create predictions table
    op.create_table('predictions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stock_id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=True),
        sa.Column('prediction_date', sa.Date(), nullable=False),
        sa.Column('target_date', sa.Date(), nullable=False),
        sa.Column('predicted_close', sa.Float(), nullable=False),
        sa.Column('predicted_high', sa.Float(), nullable=True),
        sa.Column('predicted_low', sa.Float(), nullable=True),
        sa.Column('confidence_upper', sa.Float(), nullable=True),
        sa.Column('confidence_lower', sa.Float(), nullable=True),
        sa.Column('actual_close', sa.Float(), nullable=True),
        sa.Column('prediction_error', sa.Float(), nullable=True),
        sa.Column('model_name', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['model_id'], ['model_metrics.id'], ),
        sa.ForeignKeyConstraint(['stock_id'], ['stocks.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_predictions_id'), 'predictions', ['id'], unique=False)
    op.create_index(op.f('ix_predictions_target_date'), 'predictions', ['target_date'], unique=False)
    op.create_index('ix_predictions_stock_target', 'predictions', ['stock_id', 'target_date'], unique=False)
    op.create_index('ix_predictions_model_target', 'predictions', ['model_id', 'target_date'], unique=False)
    
    # Create news_articles table
    op.create_table('news_articles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stock_symbol', sa.String(length=20), nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('url', sa.String(length=1000), nullable=True),
        sa.Column('source', sa.String(length=100), nullable=True),
        sa.Column('author', sa.String(length=200), nullable=True),
        sa.Column('published_date', sa.DateTime(), nullable=True),
        sa.Column('sentiment_score', sa.Float(), nullable=True),
        sa.Column('sentiment_label', sa.String(length=20), nullable=True),
        sa.Column('scraped_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('url')
    )
    op.create_index(op.f('ix_news_articles_id'), 'news_articles', ['id'], unique=False)
    op.create_index(op.f('ix_news_articles_stock_symbol'), 'news_articles', ['stock_symbol'], unique=False)
    op.create_index(op.f('ix_news_articles_published_date'), 'news_articles', ['published_date'], unique=False)
    op.create_index('ix_news_symbol_date', 'news_articles', ['stock_symbol', 'published_date'], unique=False)


def downgrade() -> None:
    op.drop_table('news_articles')
    op.drop_table('predictions')
    op.drop_table('model_metrics')
    op.drop_table('sentiment_analysis')
    op.drop_table('technical_indicators')
    op.drop_table('stock_prices')
    op.drop_table('stocks')
