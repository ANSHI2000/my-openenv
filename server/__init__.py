"""Healthcare Scheduling Server"""

def main() -> None:
    """Main entry point for the server"""
    import uvicorn
    from server.app import app
    
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
