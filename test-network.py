import asyncio
from fastmcp import Client

async def main():
    async with Client("server.py") as client:
        tools = await client.list_tools()
        print("Available tools:", tools)

        result = await client.call_tool("get_cell_location", {
            "mcc": 404,
            "mnc": 45,
            "lac": 1234,
            "cid": 5678901
        })
        print("\nResult:", result)

if __name__ == "__main__":
    asyncio.run(main())
