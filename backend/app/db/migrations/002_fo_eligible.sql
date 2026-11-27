-- Stocks with F&O contracts fall under the Closing Auction Session (earlier continuous-trading end)
ALTER TABLE instruments ADD COLUMN IF NOT EXISTS fo_eligible BOOLEAN NOT NULL DEFAULT FALSE;
