# ClaimSafer Premium

This is the premium version of ClaimSafer with authentication and subscription management.

## Features

### Authentication
- User registration and login with email/password
- JWT token-based authentication
- Secure password hashing with bcrypt

### Subscription Tiers
- **Basic**: 50 searches/month + 3 variations per claim
- **Premium**: Unlimited searches + all variations

### Stripe Integration
- Automatic subscription management
- Webhook handling for subscription events
- Checkout session creation for upgrades

## Environment Variables

Create a `.env` file with the following variables:

```env
# Authentication
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=postgresql://username:password@host:port/database

# Stripe
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_BASIC_PRICE_ID=price_...
STRIPE_PREMIUM_PRICE_ID=price_...
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables

3. Run the application:
```bash
uvicorn main_premium:app --reload
```

## API Endpoints

### Authentication
- `POST /api/register` - Register new user
- `POST /api/login` - Login user
- `GET /api/profile` - Get user profile (requires auth)

### Search (requires authentication)
- `POST /search-by-ingredient-premium` - Search by ingredient
- `POST /search-by-claim-premium` - Search by claim

### Stripe
- `POST /api/create-checkout-session` - Create Stripe checkout
- `POST /api/webhook/stripe` - Stripe webhook handler

## Database Schema

The application uses SQLAlchemy with the following User model:

```python
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    subscription_tier = Column(String, default="basic")
    subscription_status = Column(String, default="active")
    stripe_customer_id = Column(String, nullable=True)
    stripe_subscription_id = Column(String, nullable=True)
    searches_used = Column(Integer, default=0)
    searches_limit = Column(Integer, default=50)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
```

## Deployment

This version is designed to be deployed to Railway or similar platforms.

### Railway Deployment
1. Connect your GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically on push to main branch

### Custom Domain
- Main app: `app.claimsafer.com` (free version)
- Premium app: `admin.claimsafer.com` (paid version)

## Security Features

- Password hashing with bcrypt
- JWT token authentication
- Rate limiting based on subscription tier
- Secure Stripe webhook handling
- Input validation and sanitization 