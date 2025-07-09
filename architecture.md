# GenAI Processing Engine Architecture

```mermaid
graph TD
    subgraph User
        U1(User logs in via Web App)
        U2(User selects use case & uploads data)
    end

    subgraph Frontend (Web App)
        FE1(Collects user input: text, image, audio, etc.)
        FE2(Uploads input to Supabase Storage & DB)
    end

    subgraph Supabase
        SB1(Storage: Raw Inputs)
        SB2(Database: Metadata, Job Status)
        SB3(Storage: Processed Outputs)
    end

    subgraph Processing Engine (SkyPilot)
        direction TB
        PE1[SkyPilot YAML (per use case)]
        PE2[SkyPilot launches GPU VM]
        PE3[Processing Script (modular)]
        PE4[Common Module: Connect/Read/Save]
        PE5[Use Case Logic: Process & Generate Output]
    end

    U1 --> U2
    U2 --> FE1
    FE1 --> FE2
    FE2 --> SB1
    FE2 --> SB2

    %% Periodic/triggered processing
    SB1 -.->|Polling/Trigger| PE1
    PE1 --> PE2
    PE2 --> PE3
    PE3 --> PE4
    PE3 --> PE5
    PE4 -->|Read Input| SB1
    PE5 -->|Process| PE4
    PE4 -->|Save Output| SB3
    PE4 -->|Update Status| SB2

    %% Extensibility
    PE3 -.->|New Use Case| PE5
    PE1 -.->|New Use Case YAML| PE2

    %% Output flow
    SB3 -->|Frontend fetches output| FE1
```

---

## Key Points
- **Frontend**: Handles user authentication, input collection, and uploads to Supabase.
- **Supabase**: Stores raw inputs, job metadata/status, and processed outputs.
- **Processing Engine**:
    - Each use case has a dedicated SkyPilot YAML (for resource config) and a processing script.
    - Scripts are modular: common code for Supabase I/O, separate logic for each use case.
    - SkyPilot launches jobs on demand (periodic or triggered) to process new data.
    - Outputs and status are written back to Supabase.
- **Extensibility**: Add new use cases by creating new scripts and YAMLs, reusing common modules.
- **SkyPilot**: Manages cloud resources, job scheduling, and cost efficiency.

---

