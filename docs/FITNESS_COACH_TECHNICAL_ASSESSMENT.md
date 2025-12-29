# FitCoach AI - Technical Assessment

**Date:** 2025-12-29
**Authors:** Doug Mealing, Claude
**Status:** Assessment Complete

---

## Executive Summary

Building an AI-powered fitness coaching platform on draagon-ai is **highly feasible**. The cognitive architecture (personality, memory, beliefs, learning, curiosity) maps almost perfectly to personal training requirements.

**Build time with Claude Code:** 4-8 weeks for MVP, 12-16 weeks for full product.

---

## Why draagon-ai Is Perfect for This

| draagon-ai Feature | Fitness Trainer Application |
|-------------------|----------------------------|
| **Persona System** | Mike's coaching personality + configurable variants |
| **4-Layer Memory** | Workout history, goals, injuries, preferences |
| **Belief Reconciliation** | "I can run 5km" vs actual performance data |
| **Curiosity Engine** | "You mentioned back pain - have you tried core work?" |
| **Learning Service** | Extracts what works for each user automatically |
| **Opinion Formation** | Develops views on training approaches |
| **Multi-user Support** | Each client gets personalized coaching |
| **Credibility Tracking** | Trust self-reported metrics based on accuracy history |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FitCoach Mobile App (React Native)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ Progress     │  │ Workout     │  │ Video        │  │ Chat with   │  │
│  │ Dashboard    │  │ Logging     │  │ Library      │  │ Coach       │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ REST/WebSocket API
┌────────────────────────────────────┼────────────────────────────────────┐
│                        FitCoach Server (FastAPI)                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Fitness Domain Tools                           │   │
│  │  • log_workout(type, duration, intensity, notes)                  │   │
│  │  • get_progress(user, metric, period)                            │   │
│  │  • suggest_workout(user, available_time, equipment)              │   │
│  │  • analyze_form(video/description)                               │   │
│  │  • track_weight(user, weight, date)                              │   │
│  │  • set_goal(user, goal_type, target, deadline)                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    FitCoach Persona                               │   │
│  │  • Mike's coaching philosophy & style                             │   │
│  │  • Configurable variants (tough love vs gentle)                   │   │
│  │  • Domain-specific prompts (form cues, motivation)                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────┤
│                    draagon-ai (REUSE 90%)                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Persona    │  │  Memory     │  │  Cognition  │  │  Learning   │    │
│  │  Manager    │  │  (4-layer)  │  │  (beliefs)  │  │  Service    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
├─────────────────────────────────────────────────────────────────────────┤
│                    Infrastructure (REUSE 100%)                          │
│  Groq LLM • Qdrant Memory • FastAPI • Optional: Ollama for embeddings   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Killer Features From draagon-ai

### 1. Remembers Everything About Each Client

```
Week 1: "I have a bad knee from an old injury"
Week 8: *Coach automatically avoids high-impact exercises*
Week 12: "Can I try running?"
Coach: "Your knee has been holding up well in low-impact work.
        Let's start with walk-run intervals and see how it feels.
        Remember to ice after if there's any discomfort."
```

### 2. Learns What Works For Each Person

The Learning Service automatically extracts patterns:
- "User is more consistent with morning workouts"
- "User responds well to progressive challenges"
- "User needs recovery reminders after leg days"

### 3. Detects Inconsistencies (Belief Reconciliation)

```
User: "I've been eating clean all week"
Coach: "That's great! Though I noticed you logged a workout
        on Tuesday feeling low energy - sometimes that can
        be nutrition related. How are your protein levels?"
```

### 4. Proactive Coaching (Curiosity Engine)

```
*User hasn't logged in 3 days*
Coach: "Hey! I noticed you've been quiet. Everything okay?
        Sometimes life gets in the way - want to talk about
        a lighter schedule this week?"
```

### 5. Adapts Coaching Style Per User

Via the Persona traits system:
- **Tough love client**: Higher assertiveness (0.8), lower empathy (0.4)
- **Needs encouragement**: Lower assertiveness (0.3), higher empathy (0.9)
- **Just wants data**: Higher verbosity (0.6), lower warmth (0.3)

---

## AI vs Human Trainer Comparison

| Capability | AI Coach | Human Trainer | Winner |
|------------|----------|---------------|--------|
| **Availability** | 24/7, instant | Limited hours | AI |
| **Cost** | $20-50/month | $200-800/month | AI |
| **Consistency** | Never has bad days | Variable | AI |
| **Memory** | Perfect recall | Forgets details | AI |
| **Personalization** | Learns over time | Learns over time | Tie |
| **Form Correction** | Good with video analysis | Excellent in-person | Human |
| **Motivation** | Persistent but scripted | Genuine connection | Human |
| **Complex Assessment** | Pattern-based | Intuitive + experience | Human |
| **Hands-on Spotting** | Impossible | Essential for heavy lifts | Human |
| **Program Design** | Template-based + adaptations | Fully custom | Human (slight) |

**Verdict:** AI wins on accessibility, cost, and consistency. Human wins on complex assessment, real-time form correction, and emotional connection.

---

## What Needs to Be Built

### Already Exists (draagon-ai - 90% Reuse)

| Component | Status | Notes |
|-----------|--------|-------|
| Persona/Personality System | ✅ Ready | Define Mike's traits |
| 4-Layer Memory | ✅ Ready | Store fitness data |
| Belief Reconciliation | ✅ Ready | Track consistency |
| Curiosity Engine | ✅ Ready | Proactive check-ins |
| Learning Service | ✅ Ready | Extract patterns |
| Multi-user Support | ✅ Ready | Per-client personalization |
| LLM Integration (Groq) | ✅ Ready | Fast, cheap inference |
| FastAPI Framework | ✅ Ready | API scaffolding |

### Needs Custom Building

| Component | Effort | Description |
|-----------|--------|-------------|
| Fitness Tools | Medium | 15-20 custom tools |
| FitCoach Persona | Low | Mike's philosophy encoded |
| Mobile App | Medium-High | React Native, ~20 screens |
| Video Library | Medium | CDN + metadata |
| Workout Templates | Medium | Mike's programs digitized |
| Progress Analytics | Medium | Charts, trends |
| Form Analysis | High | Vision model integration |
| Integrations | Medium | Apple Health, Strava |
| Booking/Payments | Medium | Stripe + scheduling |

---

## Technical Implementation Plan

### Phase 1: MVP (4-6 weeks)

**Week 1-2: Core Infrastructure**
- Fork/create new project based on draagon-ai
- Define FitCoach persona with Mike's coaching philosophy
- Implement basic fitness tools (log_workout, get_progress, set_goal)
- Set up Qdrant for fitness-specific memory types

**Week 3-4: Mobile App**
- Basic chat interface with AI coach
- Workout logging UI
- Progress dashboard
- Weight tracking

**Week 5-6: Content & Polish**
- Integrate Mike's workout videos
- Program templates based on Mike's methodology
- Onboarding flow
- Basic analytics

### Phase 2: Full Product (6-10 more weeks)

- Form analysis via video upload
- Apple Health / Strava integration
- Nutrition tracking
- Social features (challenges, leaderboards)
- Booking system for Mike's 1:1 sessions
- Admin dashboard for Mike

---

## Infrastructure Costs (Estimated)

| Service | Monthly Cost | Notes |
|---------|-------------|-------|
| Groq API | $50-200 | ~$0.60/M tokens, very efficient |
| Qdrant Cloud | $25-100 | Or self-host for free |
| Video CDN | $50-200 | Cloudflare R2 or similar |
| Server | $20-50 | Railway, Fly.io, or VPS |
| App Store | $100/year | iOS + Android |
| **Total** | **$150-550/mo** | Scales with users |

---

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Liability (bad advice) | High | Medium | Clear disclaimers, conservative recommendations |
| AI hallucination | Medium | Low | Constrain to Mike's methodology |
| User churn | High | Medium | Focus on retention mechanics |
| Mike capacity | Medium | High | Start with limited premium slots |
| Competition | Medium | High | Differentiate on hybrid model |

---

## Conclusion

draagon-ai provides 90% of the cognitive infrastructure needed. The remaining work is:
1. Fitness-specific tools (~15-20)
2. Mike's persona definition
3. Mobile app UI
4. Content (videos, programs)

**Recommendation:** Proceed to business model analysis and competitive research.
