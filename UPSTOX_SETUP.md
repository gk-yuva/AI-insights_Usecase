# Upstox API Integration Setup Guide

This guide will help you set up your Upstox API credentials for the portfolio analysis system.

## Step 1: Get Upstox API Credentials

1. Go to [Upstox Developer Portal](https://upstox.com/developer/apps/)
2. Log in with your Upstox account
3. Create a new app or use an existing one
4. Note down:
   - **API Key** (Client ID)
   - **API Secret** (Client Secret)
   - **Redirect URI** (set during app creation)

## Step 2: Generate Access Token

You have two options:

### Option A: Use Upstox Login Flow (Recommended)

1. Run the token generation script:
   ```bash
   python generate_upstox_token.py
   ```

2. The script will:
   - Open your browser for Upstox login
   - Ask you to authorize the app
   - Generate an access token
   - Save it to your .env file

### Option B: Manual Token Generation

1. Visit this URL (replace `YOUR_API_KEY` and `YOUR_REDIRECT_URI`):
   ```
   https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id=YOUR_API_KEY&redirect_uri=YOUR_REDIRECT_URI
   ```

2. After authorization, you'll be redirected to your redirect URI with a `code` parameter
3. Copy the code value
4. Exchange it for an access token using the API or update AUTH_CODE in .env

## Step 3: Configure .env File

1. Copy `.env.template` to `.env`:
   ```bash
   copy .env.template .env
   ```

2. Open `.env` and fill in your credentials:
   ```
   UPSTOX_API_KEY=your_actual_api_key
   UPSTOX_API_SECRET=your_actual_secret
   UPSTOX_REDIRECT_URI=your_redirect_uri
   UPSTOX_ACCESS_TOKEN=your_access_token
   ```

## Step 4: Verify Configuration

Run the configuration checker:
```bash
python config.py
```

You should see: ✓ Upstox API credentials configured

## Step 5: Run Portfolio Analysis

```bash
python portfolio_analyzer.py
```

The system will now use Upstox API for Indian stocks instead of yfinance!

## Troubleshooting

- **Access token expired**: Upstox access tokens typically expire daily. Re-run the token generation script.
- **Invalid credentials**: Double-check your API key and secret.
- **Fallback to yfinance**: If Upstox fails, the system automatically falls back to yfinance.

## Security Notes

- ⚠️ Never commit `.env` file to version control
- ⚠️ Keep your API credentials secure
- ⚠️ Regenerate tokens if compromised
